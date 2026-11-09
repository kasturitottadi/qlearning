"""
main.py
=======
Entry point for the Q-Learning Hyper-Heuristic Cloud Workflow Scheduler.

Usage
-----
    python main.py [OPTIONS]

Options
-------
    --workflow PATH     Path to a workflow JSON/YAML file (default: synthetic).
    --tasks INT         Number of tasks in synthetic workflow (default: 15).
    --hosts INT         Number of cloud hosts (default: 5).
    --episodes INT      Number of Q-learning episodes (default: 200).
    --alpha FLOAT       Q-learning rate (default: 0.1).
    --gamma FLOAT       Discount factor (default: 0.9).
    --epsilon FLOAT     Initial exploration rate (default: 1.0).
    --seed INT          Random seed (default: 42).
    --output DIR        Output directory for CSV results (default: results/).
    --dvfs BOOL         Enable DVFS energy scaling (default: True).
    --verbose           Enable DEBUG-level logging.

Examples
--------
    # Quick test with defaults (synthetic 15-task workflow)
    python main.py

    # Custom workflow file, 300 episodes
    python main.py --workflow data/montage.json --episodes 300

    # Larger experiment
    python main.py --tasks 25 --hosts 8 --episodes 500 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Package bootstrap: add project root to sys.path so relative imports resolve
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from parsers.dag_parser import DAGParser, generate_synthetic_workflow
from core.state_evaluator import StateEvaluator
from agents.ql_agent import QLearningAgent, RewardCalculator
from simulation.engine import SimulationEngine, default_host_fleet
from utils.metrics import MetricsLogger


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-Learning Hyper-Heuristic for Cloud Workflow Scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Path to workflow JSON/YAML file. If omitted, a synthetic DAG is generated.",
    )
    parser.add_argument("--tasks", type=int, default=15, help="Tasks in synthetic workflow.")
    parser.add_argument("--hosts", type=int, default=5, help="Number of cloud hosts.")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning rate.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="Minimum exploration rate."
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.98, help="Epsilon decay per episode."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory for reports."
    )
    parser.add_argument(
        "--num-bins", type=int, default=5, help="Discrete state bins per dimension."
    )
    parser.add_argument(
        "--max-queue", type=int, default=20, help="Max queue length for normalisation."
    )
    parser.add_argument(
        "--w-energy", type=float, default=0.4, help="Reward weight for energy."
    )
    parser.add_argument(
        "--w-makespan", type=float, default=0.4, help="Reward weight for makespan."
    )
    parser.add_argument(
        "--w-load", type=float, default=0.2, help="Reward weight for load imbalance."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ql_experiment",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("Q-Learning Hyper-Heuristic Cloud Workflow Scheduler")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  Episodes  : %d", args.episodes)
    logger.info("  Hosts     : %d", args.hosts)
    logger.info("  α / γ / ε : %.2f / %.2f / %.2f", args.alpha, args.gamma, args.epsilon)
    logger.info("  Seed      : %d", args.seed)

    # ------------------------------------------------------------------
    # 1. Load / generate workflow
    # ------------------------------------------------------------------
    if args.workflow:
        logger.info("Loading workflow from: %s", args.workflow)
        workflow = DAGParser(args.workflow).parse()
    else:
        logger.info("Generating synthetic workflow with %d tasks…", args.tasks)
        workflow = generate_synthetic_workflow(num_tasks=args.tasks, seed=args.seed)

    logger.info(
        "Workflow '%s': %d tasks, %d edges",
        workflow.name,
        len(workflow.tasks),
        len(workflow.edges),
    )

    # ------------------------------------------------------------------
    # 2. Create cloud host fleet
    # ------------------------------------------------------------------
    hosts = default_host_fleet(num_hosts=args.hosts, seed=args.seed)

    # ------------------------------------------------------------------
    # 3. Instantiate state evaluator
    # ------------------------------------------------------------------
    state_evaluator = StateEvaluator(
        num_bins=args.num_bins,
        max_queue_length=args.max_queue,
    )

    # ------------------------------------------------------------------
    # 4. Instantiate Q-learning agent
    # ------------------------------------------------------------------
    reward_calc = RewardCalculator(
        w_energy=args.w_energy,
        w_makespan=args.w_makespan,
        w_load_imbalance=args.w_load,
    )

    agent = QLearningAgent(
        num_bins=args.num_bins,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        reward_calculator=reward_calc,
    )

    # ------------------------------------------------------------------
    # 5. Metrics logger
    # ------------------------------------------------------------------
    metrics_logger = MetricsLogger(
        output_dir=args.output,
        experiment_name=args.experiment_name,
    )

    # ------------------------------------------------------------------
    # 6. Run simulation
    # ------------------------------------------------------------------
    engine = SimulationEngine(
        workflow=workflow,
        hosts=hosts,
        agent=agent,
        state_evaluator=state_evaluator,
        metrics_logger=metrics_logger,
        max_episodes=args.episodes,
        seed=args.seed,
    )

    logger.info("Starting simulation…")
    engine.run()

    # ------------------------------------------------------------------
    # 7. Per-heuristic performance breakdown
    # ------------------------------------------------------------------
    print("\n  Per-Heuristic Average Performance:")
    per_h = metrics_logger.per_heuristic_averages()
    header = f"  {'Heuristic':<12} | {'Energy(kWh)':>12} | {'Makespan(s)':>11} | {'Util(%)':>8} | {'Reward':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for h_name, stats in sorted(per_h.items()):
        print(
            f"  {h_name:<12} | {stats['energy_kwh']:>12.6f} | "
            f"{stats['makespan_s']:>11.3f} | {stats['utilisation_pct']:>8.1f} | "
            f"{stats['reward']:>8.4f}"
        )

    # Save Q-table
    q_table_path = Path(args.output) / f"{args.experiment_name}_qtable.npz"
    agent.save_q_table(str(q_table_path))
    print(f"\n  Q-table saved → {q_table_path}")
    print("\n  Simulation complete. Goodbye!\n")


if __name__ == "__main__":
    main()

