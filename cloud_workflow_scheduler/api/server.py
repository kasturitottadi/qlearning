"""
api/server.py
=============
FastAPI backend for the Q-Learning Hyper-Heuristic Dashboard.

Endpoints
---------
GET  /api/status          – System state, host utilisation, last-episode metrics
POST /api/simulate        – Run N episodes with given hyper-parameters
GET  /api/results         – Historical benchmark comparisons and Q-table stats
WS   /ws/live-stream      – Real-time broadcast of per-episode telemetry

Run with:
    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap – make project root importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from parsers.dag_parser import DAGParser, generate_synthetic_workflow
from core.state_evaluator import StateEvaluator
from agents.ql_agent import HEURISTIC_NAMES, QLearningAgent, RewardCalculator
from simulation.engine import SimulationEngine, CloudHost, default_host_fleet
from utils.metrics import (
    EpisodeResult,
    MetricsLogger,
    compute_energy,
    compute_host_utilisation,
    compute_load_imbalance,
    compute_makespan,
    WS_TO_KWH,
)

logger = logging.getLogger("api.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Global simulation state (shared across requests)
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    running: bool = False
    episode: int = 0
    total_episodes: int = 0
    # Last episode metrics
    last_energy_kwh: float = 0.0
    last_makespan_s: float = 0.0
    last_utilisation_pct: float = 0.0
    last_load_imbalance: float = 0.0
    last_reward: float = 0.0
    last_heuristic: str = "—"
    # Cumulative history (capped at 1000 for memory)
    rewards: List[float] = None
    makespans: List[float] = None
    energies: List[float] = None
    heuristic_counts: Dict[str, int] = None
    host_utilisation: Dict[str, float] = None
    epsilon: float = 1.0
    # Task execution log (last 50)
    task_log: List[Dict] = None

    def __post_init__(self):
        self.rewards = []
        self.makespans = []
        self.energies = []
        self.heuristic_counts = {h: 0 for h in HEURISTIC_NAMES}
        self.host_utilisation = {}
        self.task_log = []

    def record(self, result: EpisodeResult, host_utils: Dict[str, float]) -> None:
        self.episode += 1
        self.last_energy_kwh = result.total_energy_kwh
        self.last_makespan_s = result.makespan_s
        self.last_utilisation_pct = result.avg_host_utilisation_pct
        self.last_load_imbalance = result.load_imbalance
        self.last_reward = result.reward
        self.last_heuristic = result.heuristic_used
        # Rolling history (keep last 500)
        self.rewards = (self.rewards + [result.reward])[-500:]
        self.makespans = (self.makespans + [result.makespan_s])[-500:]
        self.energies = (self.energies + [result.total_energy_kwh])[-500:]
        self.heuristic_counts[result.heuristic_used] = (
            self.heuristic_counts.get(result.heuristic_used, 0) + 1
        )
        self.host_utilisation = host_utils

    def reset(self) -> None:
        self.__init__()


SIM_STATE = SimState()
AGENT: Optional[QLearningAgent] = None
HOSTS: List[CloudHost] = []
METRICS_LOGGER: Optional[MetricsLogger] = None

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WS client connected. Total: %d", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WS client disconnected. Total: %d", len(self.active))

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(payload, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise default simulation components on startup."""
    global AGENT, HOSTS, METRICS_LOGGER
    HOSTS = default_host_fleet(num_hosts=5, seed=42)
    AGENT = QLearningAgent(num_bins=5, seed=42)
    METRICS_LOGGER = MetricsLogger(output_dir=str(ROOT / "results"), experiment_name="dashboard")
    for h in HOSTS:
        SIM_STATE.host_utilisation[h.host_id] = 0.0
    logger.info("Dashboard backend initialised with %d hosts.", len(HOSTS))
    yield


app = FastAPI(
    title="Q-Learning Scheduler Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    episodes: int = Field(default=10, ge=1, le=1000)
    alpha: float = Field(default=0.1, ge=0.001, le=1.0)
    gamma: float = Field(default=0.9, ge=0.0, le=1.0)
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.98, ge=0.8, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=0.5)
    num_hosts: int = Field(default=5, ge=2, le=10)
    workflow: str = Field(default="synthetic")   # "synthetic" | "montage" | "cybershake"
    num_tasks: int = Field(default=15, ge=5, le=50)
    w_energy: float = Field(default=0.4)
    w_makespan: float = Field(default=0.4)
    w_load: float = Field(default=0.2)
    seed: int = Field(default=42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKFLOW_MAP = {
    "montage": ROOT / "data" / "montage.json",
    "cybershake": ROOT / "data" / "cybershake.yaml",
}


def _load_workflow(req: SimulateRequest):
    if req.workflow in WORKFLOW_MAP:
        return DAGParser(WORKFLOW_MAP[req.workflow]).parse()
    return generate_synthetic_workflow(num_tasks=req.num_tasks, seed=req.seed)


def _host_util_snapshot(hosts: List[CloudHost], makespan: float) -> Dict[str, float]:
    return {
        h.host_id: round(min((h.total_busy_time_s / max(makespan, 1e-9)) * 100.0, 100.0), 1)
        for h in hosts
    }


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Return current system state, host utilisation, and last-episode metrics."""
    return {
        "running": SIM_STATE.running,
        "episode": SIM_STATE.episode,
        "total_episodes": SIM_STATE.total_episodes,
        "epsilon": round(SIM_STATE.epsilon, 4),
        "last_metrics": {
            "energy_kwh": round(SIM_STATE.last_energy_kwh, 6),
            "makespan_s": round(SIM_STATE.last_makespan_s, 3),
            "utilisation_pct": round(SIM_STATE.last_utilisation_pct, 2),
            "load_imbalance": round(SIM_STATE.last_load_imbalance, 4),
            "reward": round(SIM_STATE.last_reward, 4),
            "heuristic": SIM_STATE.last_heuristic,
        },
        "host_utilisation": SIM_STATE.host_utilisation,
        "heuristic_counts": SIM_STATE.heuristic_counts,
        "num_active_ws": len(manager.active),
    }


@app.post("/api/simulate")
async def run_simulation(req: SimulateRequest):
    """
    Launch N simulation episodes in a background task.
    Results stream over /ws/live-stream.
    """
    if SIM_STATE.running:
        return {"status": "already_running", "episode": SIM_STATE.episode}

    asyncio.create_task(_simulation_task(req))
    return {"status": "started", "episodes": req.episodes, "workflow": req.workflow}


@app.post("/api/reset")
async def reset_simulation():
    """Reset agent, hosts, and all accumulated metrics."""
    global AGENT, HOSTS, METRICS_LOGGER
    SIM_STATE.reset()
    HOSTS = default_host_fleet(num_hosts=5, seed=42)
    AGENT = QLearningAgent(num_bins=5, seed=42)
    METRICS_LOGGER = MetricsLogger(output_dir=str(ROOT / "results"), experiment_name="dashboard")
    for h in HOSTS:
        SIM_STATE.host_utilisation[h.host_id] = 0.0
    await manager.broadcast({"type": "reset", "message": "Simulation reset."})
    return {"status": "reset"}


@app.get("/api/results")
async def get_results():
    """Historical benchmark comparisons, Q-table stats, and heuristic counters."""
    q_states = len(AGENT.q_table) if AGENT else 0
    best_actions = {}
    if AGENT:
        for state, q_vals in list(AGENT.q_table.items())[:20]:
            best_a = int(np.argmax(q_vals))
            best_actions[str(state)] = {
                "best_heuristic": HEURISTIC_NAMES[best_a],
                "q_values": {HEURISTIC_NAMES[i]: round(float(q_vals[i]), 4) for i in range(4)},
            }

    # Per-heuristic averages from logger
    per_h_avgs = METRICS_LOGGER.per_heuristic_averages() if METRICS_LOGGER else {}

    convergence = AGENT.convergence_summary() if AGENT else {}

    return {
        "rewards": SIM_STATE.rewards[-200:],
        "makespans": SIM_STATE.makespans[-200:],
        "energies": SIM_STATE.energies[-200:],
        "heuristic_counts": SIM_STATE.heuristic_counts,
        "q_table": {
            "num_states_visited": q_states,
            "sample_entries": best_actions,
        },
        "per_heuristic_averages": per_h_avgs,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/live-stream")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current state immediately on connect
        await ws.send_text(json.dumps({
            "type": "init",
            "episode": SIM_STATE.episode,
            "rewards": SIM_STATE.rewards[-100:],
            "makespans": SIM_STATE.makespans[-100:],
            "heuristic_counts": SIM_STATE.heuristic_counts,
            "host_utilisation": SIM_STATE.host_utilisation,
        }, default=str))
        while True:
            await ws.receive_text()   # Keep alive / handle client pings
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Background simulation task (async, broadcasts per episode)
# ---------------------------------------------------------------------------

async def _simulation_task(req: SimulateRequest) -> None:
    global AGENT, HOSTS, METRICS_LOGGER
    SIM_STATE.running = True
    SIM_STATE.total_episodes += req.episodes

    # Build components
    workflow = _load_workflow(req)
    hosts = default_host_fleet(num_hosts=req.num_hosts, seed=req.seed)
    HOSTS = hosts

    state_evaluator = StateEvaluator(num_bins=5, max_queue_length=20)
    reward_calc = RewardCalculator(
        w_energy=req.w_energy,
        w_makespan=req.w_makespan,
        w_load_imbalance=req.w_load,
    )

    # Reuse existing agent (transfer learning) or create fresh
    if AGENT is None or req.epsilon >= 0.99:
        AGENT = QLearningAgent(
            num_bins=5,
            alpha=req.alpha,
            gamma=req.gamma,
            epsilon_start=req.epsilon,
            epsilon_end=req.epsilon_end,
            epsilon_decay=req.epsilon_decay,
            seed=req.seed,
            reward_calculator=reward_calc,
        )
    else:
        AGENT.alpha = req.alpha
        AGENT.gamma = req.gamma
        AGENT.reward_calc = reward_calc
        AGENT.epsilon_decay = req.epsilon_decay

    metrics_logger = MetricsLogger(
        output_dir=str(ROOT / "results"), experiment_name="dashboard"
    )
    METRICS_LOGGER = metrics_logger

    engine = SimulationEngine(
        workflow=workflow,
        hosts=hosts,
        agent=AGENT,
        state_evaluator=state_evaluator,
        metrics_logger=metrics_logger,
        max_episodes=req.episodes,
        seed=req.seed,
    )

    # Calibrate once
    baseline_e, baseline_m = engine._calibrate()
    AGENT.reward_calc = RewardCalculator(
        w_energy=req.w_energy,
        w_makespan=req.w_makespan,
        w_load_imbalance=req.w_load,
        baseline_energy=baseline_e,
        baseline_makespan=baseline_m,
    )

    for ep in range(req.episodes):
        if not SIM_STATE.running:
            break

        t0 = time.perf_counter()
        engine._reset_hosts()

        # State evaluation
        snapshots = [h.snapshot() for h in hosts]
        ready_q = len(workflow.entry_tasks)
        state = state_evaluator.evaluate(snapshots, ready_q)
        state_metrics = state_evaluator.continuous_metrics(snapshots, ready_q)

        # Agent action
        action = AGENT.select_action(state)
        heuristic_name = HEURISTIC_NAMES[action]

        # Execute heuristic
        task_execs = engine._apply_heuristic(action)

        # Metrics
        result = engine._compute_episode_metrics(
            episode_id=SIM_STATE.episode,
            heuristic_name=heuristic_name,
            task_executions=task_execs,
        )

        # Update hosts
        engine._update_host_state(task_execs)
        host_utils = _host_util_snapshot(hosts, result.makespan_s)

        # Next state + Q-update
        next_snapshots = [h.snapshot() for h in hosts]
        next_state = state_evaluator.evaluate(next_snapshots, 0)
        td_error = AGENT.update(
            state=state, action=action,
            reward=result.reward, next_state=next_state, done=True,
        )
        AGENT.end_episode(result.reward)

        # Update global state
        SIM_STATE.record(result, host_utils)
        SIM_STATE.epsilon = AGENT.epsilon
        metrics_logger.record(result)

        # Build task log entries
        task_log_entries = [
            {
                "task_id": te.task_id,
                "host_id": te.host_id,
                "start": round(te.start_time, 3),
                "finish": round(te.finish_time, 3),
                "energy_ws": round(te.energy_ws, 4),
            }
            for te in task_execs[:20]  # Cap broadcast size
        ]

        # Broadcast tick
        payload = {
            "type": "episode_tick",
            "episode": SIM_STATE.episode,
            "heuristic": heuristic_name,
            "metrics": {
                "energy_kwh": round(result.total_energy_kwh, 6),
                "makespan_s": round(result.makespan_s, 3),
                "utilisation_pct": round(result.avg_host_utilisation_pct, 2),
                "load_imbalance": round(result.load_imbalance, 4),
                "reward": round(result.reward, 4),
            },
            "state_vector": list(state),
            "state_metrics": {k: round(v, 3) for k, v in state_metrics.items()},
            "td_error": round(float(td_error), 5),
            "epsilon": round(AGENT.epsilon, 4),
            "q_values": {
                HEURISTIC_NAMES[i]: round(float(AGENT.q_table[state][i]), 4)
                for i in range(4)
            },
            "host_utilisation": host_utils,
            "heuristic_counts": SIM_STATE.heuristic_counts,
            "task_allocations": task_log_entries,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        await manager.broadcast(payload)
        await asyncio.sleep(0)   # yield to event loop

    # Export CSV
    metrics_logger.export_csv()

    SIM_STATE.running = False
    await manager.broadcast({
        "type": "simulation_complete",
        "total_episodes": SIM_STATE.episode,
        "convergence": AGENT.convergence_summary(),
    })
    logger.info("Simulation task complete. Total episodes so far: %d", SIM_STATE.episode)

