"""
utils/metrics.py
================
Performance metrics collection, computation, and reporting.

Tracks per-episode results (energy, makespan, host utilisation, load imbalance)
and provides helpers to export CSV reports and print convergence summaries.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# kWh conversion factor (Watt·second → kWh)
WS_TO_KWH: float = 1.0 / 3_600_000.0


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Container for all metrics collected during one simulation episode."""

    episode_id: int
    heuristic_used: str
    total_energy_kwh: float
    makespan_s: float
    avg_host_utilisation_pct: float   # 0–100
    load_imbalance: float             # 0–1 (0 = perfectly balanced)
    reward: float
    num_tasks: int
    num_hosts: int
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )

    def as_dict(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "heuristic_used": self.heuristic_used,
            "total_energy_kwh": round(self.total_energy_kwh, 6),
            "makespan_s": round(self.makespan_s, 4),
            "avg_host_utilisation_pct": round(self.avg_host_utilisation_pct, 2),
            "load_imbalance": round(self.load_imbalance, 4),
            "reward": round(self.reward, 4),
            "num_tasks": self.num_tasks,
            "num_hosts": self.num_hosts,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_makespan(task_finish_times: List[float]) -> float:
    """
    Compute the workflow makespan (maximum finish time).

    Parameters
    ----------
    task_finish_times : list[float]
        Finish times of all tasks in seconds.

    Returns
    -------
    float
        Makespan in seconds.
    """
    return max(task_finish_times) if task_finish_times else 0.0


def compute_host_utilisation(
    host_busy_times: Dict[str, float],
    makespan_s: float,
) -> Dict[str, float]:
    """
    Compute CPU utilisation percentage for each host.

    Parameters
    ----------
    host_busy_times : dict[str, float]
        Total busy time (seconds) per host_id.
    makespan_s : float
        Workflow makespan (denominator).

    Returns
    -------
    dict[str, float]
        host_id → utilisation percentage (0–100).
    """
    if makespan_s == 0:
        return {h: 0.0 for h in host_busy_times}
    return {
        h: min((bt / makespan_s) * 100.0, 100.0)
        for h, bt in host_busy_times.items()
    }


def compute_load_imbalance(host_utilisation_pct: Dict[str, float]) -> float:
    """
    Compute the load-imbalance index (Coefficient of Variation of utilisations).

    Returns
    -------
    float
        Load imbalance in [0, 1] (0 = perfectly balanced).
    """
    utils = list(host_utilisation_pct.values())
    if not utils:
        return 0.0
    mean = float(np.mean(utils))
    if mean == 0:
        return 0.0
    std = float(np.std(utils))
    # CV normalised to [0, 1] using the fact that max CV ≈ sqrt(n-1)/sqrt(n) ≈ 1
    cv = std / mean
    return min(cv, 1.0)


def compute_energy(
    host_busy_times: Dict[str, float],
    host_mips: Dict[str, float],
    p_static_w: float = 50.0,
    p_dynamic_coeff: float = 100.0,
    dvfs_enabled: bool = True,
) -> float:
    """
    Estimate total energy consumption using a linear power model with optional DVFS.

    Power model::

        P(u) = P_static + P_dynamic * u        (without DVFS)
        P(u) = P_static + P_dynamic * u^3      (with DVFS – cubic voltage scaling)

    where u = CPU utilisation fraction.

    Parameters
    ----------
    host_busy_times : dict[str, float]
        Seconds each host was busy executing tasks.
    host_mips : dict[str, float]
        Nominal MIPS capacity per host (used to infer utilisation).
    p_static_w : float
        Static (idle) power draw in Watts (default 50 W).
    p_dynamic_coeff : float
        Dynamic power coefficient in Watts at full load (default 100 W).
    dvfs_enabled : bool
        Apply cubic DVFS scaling to dynamic power (default True).

    Returns
    -------
    float
        Total energy in kWh.
    """
    # Determine makespan from host busy times as an approximation
    total_energy_ws = 0.0
    for host_id, busy_s in host_busy_times.items():
        # CPU utilisation approximation: busy_time / makespan (capped at 1)
        # Here we use busy_s directly as the active duration
        utilisation = min(1.0, busy_s / max(sum(host_busy_times.values()), 1e-9))
        if dvfs_enabled:
            p_dynamic = p_dynamic_coeff * (utilisation ** 3)
        else:
            p_dynamic = p_dynamic_coeff * utilisation
        total_power_w = p_static_w + p_dynamic
        total_energy_ws += total_power_w * busy_s

    return total_energy_ws * WS_TO_KWH


# ---------------------------------------------------------------------------
# Metrics logger / reporter
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Collects episode results and exports them to CSV / console summaries.

    Parameters
    ----------
    output_dir : str
        Directory where CSV reports are written.
    experiment_name : str
        Prefix used for output file names.
    """

    def __init__(
        self,
        output_dir: str = "results",
        experiment_name: str = "ql_experiment",
    ) -> None:
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.results: List[EpisodeResult] = []
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, result: EpisodeResult) -> None:
        """Add a completed episode result to the log."""
        self.results.append(result)
        logger.info(
            "Episode %d | heuristic=%-10s | energy=%.5f kWh | "
            "makespan=%.2f s | util=%.1f%% | LI=%.3f | reward=%.4f",
            result.episode_id,
            result.heuristic_used,
            result.total_energy_kwh,
            result.makespan_s,
            result.avg_host_utilisation_pct,
            result.load_imbalance,
            result.reward,
        )

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_csv(self) -> str:
        """
        Write all recorded results to a CSV file.

        Returns
        -------
        str
            Path of the written CSV file.
        """
        if not self.results:
            logger.warning("No results to export.")
            return ""

        filepath = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_results.csv",
        )
        fieldnames = list(self.results[0].as_dict().keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(r.as_dict())

        logger.info("Results exported to %s", filepath)
        return filepath

    def print_summary(self) -> None:
        """Print a human-readable convergence / performance summary to stdout."""
        if not self.results:
            print("No results to summarise.")
            return

        energies = [r.total_energy_kwh for r in self.results]
        makespans = [r.makespan_s for r in self.results]
        utils = [r.avg_host_utilisation_pct for r in self.results]
        rewards = [r.reward for r in self.results]

        # Per-heuristic breakdown
        from collections import defaultdict
        heuristic_counts: Dict[str, int] = defaultdict(int)
        for r in self.results:
            heuristic_counts[r.heuristic_used] += 1

        separator = "=" * 68
        print(f"\n{separator}")
        print("  Q-LEARNING HYPER-HEURISTIC — SIMULATION SUMMARY")
        print(separator)
        print(f"  Total Episodes     : {len(self.results)}")
        print(f"  Energy (kWh)       : mean={np.mean(energies):.5f}  "
              f"min={np.min(energies):.5f}  max={np.max(energies):.5f}")
        print(f"  Makespan (s)       : mean={np.mean(makespans):.2f}  "
              f"min={np.min(makespans):.2f}  max={np.max(makespans):.2f}")
        print(f"  Host Util (%)      : mean={np.mean(utils):.1f}  "
              f"min={np.min(utils):.1f}  max={np.max(utils):.1f}")
        print(f"  Reward             : mean={np.mean(rewards):.4f}  "
              f"last-10={np.mean(rewards[-10:]):.4f}")
        print()
        print("  Heuristic Selection Breakdown:")
        for h, cnt in sorted(heuristic_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / len(self.results)
            print(f"    {h:<12s} : {cnt:4d} episodes  ({pct:.1f}%)")
        print(separator)

    def per_heuristic_averages(self) -> Dict[str, Dict[str, float]]:
        """
        Compute average metrics grouped by heuristic.

        Returns
        -------
        dict[str, dict[str, float]]
            Heuristic name → {energy_kwh, makespan_s, utilisation_pct, reward}.
        """
        from collections import defaultdict
        groups: Dict[str, List[EpisodeResult]] = defaultdict(list)
        for r in self.results:
            groups[r.heuristic_used].append(r)

        averages: Dict[str, Dict[str, float]] = {}
        for hname, eps in groups.items():
            averages[hname] = {
                "energy_kwh": float(np.mean([e.total_energy_kwh for e in eps])),
                "makespan_s": float(np.mean([e.makespan_s for e in eps])),
                "utilisation_pct": float(np.mean([e.avg_host_utilisation_pct for e in eps])),
                "reward": float(np.mean([e.reward for e in eps])),
            }
        return averages

