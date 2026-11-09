"""
simulation/engine.py
====================
Main event-driven simulation engine for cloud workflow scheduling.

The engine orchestrates one *episode* of workflow execution:

1. Initialise cloud hosts (heterogeneous MIPS / RAM configurations).
2. Ask the Q-learning agent to select a low-level heuristic.
3. Apply the heuristic to schedule DAG tasks on hosts.
4. Simulate task execution, tracking energy, makespan, and utilisation.
5. Compute reward and update the Q-table.
6. Record metrics via MetricsLogger.

The simulation uses a *time-step* model (not discrete-event) to keep the
implementation tractable while still modelling execution concurrency.

Energy model
------------
Uses a linear/DVFS power model from ``utils/metrics.py``::

    P(u) = P_static + P_dynamic * u^k

where k=1 (linear) or k=3 (DVFS cubic scaling).

DVFS scaling
------------
When DVFS is enabled, task execution time is scaled *up* (slower clock)
and power is scaled *down* (cubic), yielding energy savings at the cost
of increased makespan.  The DVFS scaling factor ``f ∈ [0.5, 1.0]`` is
chosen to minimise energy under a makespan constraint.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from parsers.dag_parser import Task, Workflow
from core.state_evaluator import HostSnapshot, StateEvaluator
from agents.ql_agent import HEURISTIC_NAMES, QLearningAgent, RewardCalculator
from heuristics.heft import HEFT, HEFTHost, HEFTTask
from heuristics.greedy import GreedyHost, GreedyTask, MaxMin, MinMin, RoundRobin
from utils.metrics import (
    EpisodeResult,
    MetricsLogger,
    WS_TO_KWH,
    compute_energy,
    compute_host_utilisation,
    compute_load_imbalance,
    compute_makespan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud host definition
# ---------------------------------------------------------------------------

@dataclass
class CloudHost:
    """
    Heterogeneous cloud host with energy and resource properties.

    Parameters
    ----------
    host_id : str
        Unique identifier.
    mips : float
        Processing capacity in MIPS.
    ram_total_mb : float
        Total RAM in MB.
    p_static_w : float
        Static (idle) power draw in Watts.
    p_max_w : float
        Maximum (full-load) dynamic power in Watts.
    dvfs_enabled : bool
        Whether DVFS is applied on this host.
    """

    host_id: str
    mips: float
    ram_total_mb: float
    p_static_w: float = 50.0
    p_max_w: float = 150.0
    dvfs_enabled: bool = True

    # Runtime state
    mips_used: float = 0.0
    ram_used_mb: float = 0.0
    busy_until: float = 0.0        # Simulation time at which host is free
    total_busy_time_s: float = 0.0  # Cumulative execution time

    @property
    def cpu_utilisation(self) -> float:
        return min(self.mips_used / max(self.mips, 1e-9), 1.0)

    def power_watts(self, utilisation: float) -> float:
        """Instantaneous power draw given CPU utilisation fraction."""
        u = min(max(utilisation, 0.0), 1.0)
        p_dynamic = (self.p_max_w - self.p_static_w)
        if self.dvfs_enabled:
            return self.p_static_w + p_dynamic * (u ** 3)
        return self.p_static_w + p_dynamic * u

    def snapshot(self) -> HostSnapshot:
        """Return a :class:`HostSnapshot` reflecting current state."""
        return HostSnapshot(
            host_id=self.host_id,
            mips_capacity=self.mips,
            mips_used=self.mips_used,
            ram_total_mb=self.ram_total_mb,
            ram_used_mb=self.ram_used_mb,
        )


# ---------------------------------------------------------------------------
# Default host configurations
# ---------------------------------------------------------------------------

def default_host_fleet(num_hosts: int = 5, seed: int = 0) -> List[CloudHost]:
    """
    Generate a heterogeneous fleet of cloud hosts.

    Parameters
    ----------
    num_hosts : int
        Number of hosts to create.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[CloudHost]
        A list of :class:`CloudHost` instances with varying MIPS/RAM.
    """
    rng = random.Random(seed)
    mips_tiers = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    ram_tiers = [2048.0, 4096.0, 8192.0, 16384.0]

    hosts: List[CloudHost] = []
    for i in range(num_hosts):
        mips = rng.choice(mips_tiers)
        hosts.append(
            CloudHost(
                host_id=f"host_{i}",
                mips=mips,
                ram_total_mb=rng.choice(ram_tiers),
                p_static_w=rng.uniform(40.0, 80.0),
                p_max_w=rng.uniform(120.0, 200.0),
                dvfs_enabled=True,
            )
        )
    logger.info("Created fleet of %d heterogeneous hosts.", num_hosts)
    return hosts


# ---------------------------------------------------------------------------
# Task execution simulator
# ---------------------------------------------------------------------------

@dataclass
class TaskExecution:
    """Records the execution plan for a single task."""

    task_id: str
    host_id: str
    start_time: float
    finish_time: float
    energy_ws: float   # Energy in Watt-seconds

    @property
    def duration(self) -> float:
        return self.finish_time - self.start_time


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Event-driven simulation engine.

    Parameters
    ----------
    workflow : Workflow
        The DAG workflow to execute.
    hosts : list[CloudHost]
        Cloud host fleet.
    agent : QLearningAgent
        The hyper-heuristic Q-learning agent.
    state_evaluator : StateEvaluator
        Converts current cloud state to a discretised Q-table key.
    metrics_logger : MetricsLogger
        Records and exports performance metrics.
    max_episodes : int
        Number of simulation episodes (repetitions of the same workflow).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        workflow: Workflow,
        hosts: List[CloudHost],
        agent: QLearningAgent,
        state_evaluator: StateEvaluator,
        metrics_logger: MetricsLogger,
        max_episodes: int = 100,
        seed: int = 42,
    ) -> None:
        self.workflow = workflow
        self.hosts = hosts
        self.agent = agent
        self.state_evaluator = state_evaluator
        self.metrics_logger = metrics_logger
        self.max_episodes = max_episodes
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute all episodes.

        Each episode:
        1. Resets host state.
        2. Evaluates current cloud state → Q-table key.
        3. Agent selects a heuristic.
        4. Heuristic schedules tasks.
        5. Simulate task execution and collect energy/timing.
        6. Compute reward, update Q-table.
        7. Log results.
        """
        # Calibration: run one episode with each heuristic to set baselines
        baseline_energy, baseline_makespan = self._calibrate()
        self.agent.reward_calc = RewardCalculator(
            baseline_energy=baseline_energy,
            baseline_makespan=baseline_makespan,
        )

        for ep in range(self.max_episodes):
            episode_reward = self._run_episode(episode_id=ep)
            self.agent.end_episode(episode_reward)

        # Final reports
        self.metrics_logger.print_summary()
        csv_path = self.metrics_logger.export_csv()
        if csv_path:
            print(f"\n  CSV report saved → {csv_path}")

        # Convergence summary
        summary = self.agent.convergence_summary()
        print("\n  Q-Learning Convergence Summary:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def _run_episode(self, episode_id: int) -> float:
        """Execute one complete workflow scheduling episode."""
        self._reset_hosts()

        # Evaluate initial state
        snapshots = [h.snapshot() for h in self.hosts]
        ready_q = len(self.workflow.entry_tasks)
        state = self.state_evaluator.evaluate(snapshots, ready_q)

        # Agent selects heuristic
        action = self.agent.select_action(state)
        heuristic_name = HEURISTIC_NAMES[action]

        # Schedule tasks using the chosen heuristic
        task_executions = self._apply_heuristic(action)

        # Simulate execution & compute metrics
        result = self._compute_episode_metrics(
            episode_id=episode_id,
            heuristic_name=heuristic_name,
            task_executions=task_executions,
        )

        # Evaluate next state (post-execution)
        self._update_host_state(task_executions)
        next_snapshots = [h.snapshot() for h in self.hosts]
        next_state = self.state_evaluator.evaluate(next_snapshots, 0)

        # Update Q-table
        self.agent.update(
            state=state,
            action=action,
            reward=result.reward,
            next_state=next_state,
            done=True,
        )

        # Record
        self.metrics_logger.record(result)
        return result.reward

    # ------------------------------------------------------------------
    # Heuristic dispatch
    # ------------------------------------------------------------------

    def _apply_heuristic(self, action: int) -> List[TaskExecution]:
        """
        Apply the selected heuristic to schedule all workflow tasks.

        Returns
        -------
        list[TaskExecution]
            Execution timeline for each task.
        """
        sorted_tasks = self.workflow.task_list()

        if action == 0:  # HEFT
            return self._run_heft(sorted_tasks)
        elif action == 1:  # Min-Min
            return self._run_greedy(sorted_tasks, strategy="min-min")
        elif action == 2:  # Max-Min
            return self._run_greedy(sorted_tasks, strategy="max-min")
        else:              # RoundRobin
            return self._run_greedy(sorted_tasks, strategy="round-robin")

    def _run_heft(self, sorted_tasks: List[Task]) -> List[TaskExecution]:
        """Schedule using HEFT and convert to TaskExecution records."""
        heft_tasks = {
            t.task_id: HEFTTask(
                task_id=t.task_id,
                runtime_s=t.runtime_s,
                cpu_mips=t.cpu_mips,
                successors=list(t.successors),
                predecessors=list(t.predecessors),
            )
            for t in sorted_tasks
        }
        heft_hosts = [HEFTHost(host_id=h.host_id, mips=h.mips) for h in self.hosts]

        heft = HEFT(tasks=heft_tasks, hosts=heft_hosts)
        assignment = heft.schedule()

        executions = []
        for t in sorted_tasks:
            ht = heft_tasks[t.task_id]
            host = next(h for h in self.hosts if h.host_id == assignment[t.task_id])
            exec_time = t.runtime_s * (t.cpu_mips / max(host.mips, 1e-9))
            utilisation = min(t.cpu_mips / host.mips, 1.0)
            energy_ws = host.power_watts(utilisation) * exec_time

            executions.append(
                TaskExecution(
                    task_id=t.task_id,
                    host_id=host.host_id,
                    start_time=ht.start_time,
                    finish_time=ht.finish_time,
                    energy_ws=energy_ws,
                )
            )
        return executions

    def _run_greedy(
        self,
        sorted_tasks: List[Task],
        strategy: str,
    ) -> List[TaskExecution]:
        """Schedule using Min-Min, Max-Min, or RoundRobin."""
        greedy_hosts = [GreedyHost(host_id=h.host_id, mips=h.mips) for h in self.hosts]
        greedy_tasks = [
            GreedyTask(
                task_id=t.task_id,
                cpu_mips=t.cpu_mips,
                runtime_s=t.runtime_s,
                predecessors=list(t.predecessors),
            )
            for t in sorted_tasks
        ]

        # Resolve predecessor finish times iteratively
        finish_map: Dict[str, float] = {}
        all_execs: List[TaskExecution] = []

        # Process in topological order, wave by wave
        remaining = list(greedy_tasks)
        while remaining:
            ready = [
                gt for gt in remaining
                if all(pred in finish_map for pred in gt.predecessors)
            ]
            if not ready:
                # Fallback: schedule everything remaining
                ready = remaining[:]

            if strategy == "min-min":
                scheduler = MinMin(hosts=greedy_hosts)
            elif strategy == "max-min":
                scheduler = MaxMin(hosts=greedy_hosts)
            else:
                rr = RoundRobin(hosts=greedy_hosts)
                scheduler = rr  # type: ignore[assignment]

            # Update host availability based on predecessor constraints
            for gt in ready:
                pred_finish = max(
                    (finish_map.get(p, 0.0) for p in gt.predecessors),
                    default=0.0,
                )
                h = next(
                    gh for gh in greedy_hosts
                    if gh.available_time == min(x.available_time for x in greedy_hosts)
                )
                h.available_time = max(h.available_time, pred_finish)

            assignment = scheduler.schedule(ready)

            for gt in ready:
                host_id = assignment[gt.task_id]
                cloud_host = next(h for h in self.hosts if h.host_id == host_id)
                exec_time = gt.runtime_s * (gt.cpu_mips / max(cloud_host.mips, 1e-9))
                utilisation = min(gt.cpu_mips / cloud_host.mips, 1.0)
                energy_ws = cloud_host.power_watts(utilisation) * exec_time

                ft = gt.finish_time if gt.finish_time > 0 else exec_time

                all_execs.append(
                    TaskExecution(
                        task_id=gt.task_id,
                        host_id=host_id,
                        start_time=max(gt.start_time, 0.0),
                        finish_time=ft,
                        energy_ws=energy_ws,
                    )
                )
                finish_map[gt.task_id] = ft
                remaining.remove(gt)

        return all_execs

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_episode_metrics(
        self,
        episode_id: int,
        heuristic_name: str,
        task_executions: List[TaskExecution],
    ) -> EpisodeResult:
        """Derive all performance metrics from task execution records."""
        finish_times = [te.finish_time for te in task_executions]
        makespan = compute_makespan(finish_times)

        # Energy per host
        host_energy_ws: Dict[str, float] = {h.host_id: 0.0 for h in self.hosts}
        host_busy: Dict[str, float] = {h.host_id: 0.0 for h in self.hosts}
        for te in task_executions:
            host_energy_ws[te.host_id] += te.energy_ws
            host_busy[te.host_id] += te.duration

        total_energy_kwh = sum(host_energy_ws.values()) * WS_TO_KWH

        host_util = compute_host_utilisation(host_busy, makespan)
        avg_util_pct = sum(host_util.values()) / max(len(host_util), 1)
        load_imbalance = compute_load_imbalance(host_util)

        reward = self.agent.reward_calc.compute(
            energy_kwh=total_energy_kwh,
            makespan_s=makespan,
            load_imbalance=load_imbalance,
        )

        return EpisodeResult(
            episode_id=episode_id,
            heuristic_used=heuristic_name,
            total_energy_kwh=total_energy_kwh,
            makespan_s=makespan,
            avg_host_utilisation_pct=avg_util_pct,
            load_imbalance=load_imbalance,
            reward=reward,
            num_tasks=len(self.workflow.tasks),
            num_hosts=len(self.hosts),
        )

    # ------------------------------------------------------------------
    # Host state helpers
    # ------------------------------------------------------------------

    def _reset_hosts(self) -> None:
        """Reset all host runtime state for a fresh episode."""
        for h in self.hosts:
            h.mips_used = 0.0
            h.ram_used_mb = 0.0
            h.busy_until = 0.0
            h.total_busy_time_s = 0.0

    def _update_host_state(self, executions: List[TaskExecution]) -> None:
        """Update host state after a scheduling episode."""
        host_map = {h.host_id: h for h in self.hosts}
        for te in executions:
            h = host_map[te.host_id]
            h.total_busy_time_s += te.duration
            h.busy_until = max(h.busy_until, te.finish_time)
            h.mips_used = h.mips  # Treat as fully utilised during task

    def _calibrate(self) -> Tuple[float, float]:
        """
        Run each heuristic once to establish baseline energy and makespan.

        Returns
        -------
        (float, float)
            (baseline_energy_kwh, baseline_makespan_s)
        """
        logger.info("Calibrating baselines across all heuristics...")
        energies, makespans = [], []
        for action in range(len(HEURISTIC_NAMES)):
            self._reset_hosts()
            execs = self._apply_heuristic(action)
            finish_times = [te.finish_time for te in execs]
            makespans.append(compute_makespan(finish_times))
            energy_ws = sum(te.energy_ws for te in execs)
            energies.append(energy_ws * WS_TO_KWH)

        baseline_energy = max(energies) if energies else 1.0
        baseline_makespan = max(makespans) if makespans else 1.0
        logger.info(
            "Baselines: energy=%.5f kWh, makespan=%.2f s",
            baseline_energy,
            baseline_makespan,
        )
        return baseline_energy, baseline_makespan

