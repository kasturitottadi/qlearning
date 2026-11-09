"""
core/state_evaluator.py
=======================
Real-time cloud state monitor and state-vector builder for the Q-learning agent.

The evaluator inspects the current simulation snapshot and returns a *discretised*
state tuple that indexes into the Q-table.

State dimensions (each discretised into N bins)
------------------------------------------------
1. avg_cpu_util       – Average CPU utilisation across all hosts (0–1).
2. memory_pressure    – Fraction of hosts whose RAM utilisation > 80 %.
3. queue_length_norm  – Normalised ready-task queue length (0–1).
4. host_heterogeneity – Coefficient of Variation of host MIPS capacities.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Host snapshot
# ---------------------------------------------------------------------------

@dataclass
class HostSnapshot:
    """Instantaneous snapshot of a single cloud host's resource usage."""

    host_id: str
    mips_capacity: float    # Total MIPS
    mips_used: float        # Currently consumed MIPS
    ram_total_mb: float     # Total RAM
    ram_used_mb: float      # Currently used RAM

    @property
    def cpu_utilisation(self) -> float:
        """CPU utilisation fraction in [0, 1]."""
        if self.mips_capacity == 0:
            return 0.0
        return min(self.mips_used / self.mips_capacity, 1.0)

    @property
    def ram_utilisation(self) -> float:
        """RAM utilisation fraction in [0, 1]."""
        if self.ram_total_mb == 0:
            return 0.0
        return min(self.ram_used_mb / self.ram_total_mb, 1.0)


# ---------------------------------------------------------------------------
# State evaluator
# ---------------------------------------------------------------------------

class StateEvaluator:
    """
    Monitors cloud state and returns a discretised state tuple for Q-learning.

    Parameters
    ----------
    num_bins : int
        Number of discrete bins for each state dimension (default: 5).
    max_queue_length : int
        Maximum expected ready-task queue length (used for normalisation).
    """

    def __init__(self, num_bins: int = 5, max_queue_length: int = 20) -> None:
        self.num_bins = num_bins
        self.max_queue_length = max_queue_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        host_snapshots: List[HostSnapshot],
        ready_queue_length: int,
    ) -> Tuple[int, int, int, int]:
        """
        Compute and discretise the current cloud state.

        Parameters
        ----------
        host_snapshots : list[HostSnapshot]
            One snapshot per active host.
        ready_queue_length : int
            Number of tasks currently ready to be scheduled.

        Returns
        -------
        tuple[int, int, int, int]
            Discretised state vector ``(avg_cpu_bin, mem_pressure_bin,
            queue_bin, heterogeneity_bin)``.
        """
        if not host_snapshots:
            logger.warning("No host snapshots provided; returning zero state.")
            return (0, 0, 0, 0)

        avg_cpu = self._avg_cpu_util(host_snapshots)
        mem_pressure = self._memory_pressure(host_snapshots)
        queue_norm = self._queue_norm(ready_queue_length)
        heterogeneity = self._host_heterogeneity(host_snapshots)

        state = (
            self._discretise(avg_cpu),
            self._discretise(mem_pressure),
            self._discretise(queue_norm),
            self._discretise(heterogeneity),
        )

        logger.debug(
            "State vector (continuous): cpu=%.3f mem=%.3f queue=%.3f het=%.3f → %s",
            avg_cpu,
            mem_pressure,
            queue_norm,
            heterogeneity,
            state,
        )
        return state

    def state_space_size(self) -> int:
        """Total number of discrete states in the Q-table."""
        return self.num_bins ** 4

    def continuous_metrics(
        self,
        host_snapshots: List[HostSnapshot],
        ready_queue_length: int,
    ) -> Dict[str, float]:
        """
        Return raw (continuous) metric values for logging / debugging.

        Returns
        -------
        dict
            Keys: ``avg_cpu_util``, ``memory_pressure``,
            ``queue_length_norm``, ``host_heterogeneity``.
        """
        return {
            "avg_cpu_util": self._avg_cpu_util(host_snapshots),
            "memory_pressure": self._memory_pressure(host_snapshots),
            "queue_length_norm": self._queue_norm(ready_queue_length),
            "host_heterogeneity": self._host_heterogeneity(host_snapshots),
        }

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_cpu_util(snapshots: List[HostSnapshot]) -> float:
        """Mean CPU utilisation across all hosts (0–1)."""
        return sum(s.cpu_utilisation for s in snapshots) / len(snapshots)

    @staticmethod
    def _memory_pressure(snapshots: List[HostSnapshot]) -> float:
        """Fraction of hosts whose RAM utilisation exceeds 80 %."""
        overloaded = sum(1 for s in snapshots if s.ram_utilisation > 0.80)
        return overloaded / len(snapshots)

    def _queue_norm(self, queue_length: int) -> float:
        """Normalised ready-task queue length clamped to [0, 1]."""
        return min(queue_length / max(self.max_queue_length, 1), 1.0)

    @staticmethod
    def _host_heterogeneity(snapshots: List[HostSnapshot]) -> float:
        """
        Coefficient of Variation (CV) of host MIPS capacities, clamped to [0, 1].

        CV = std / mean.  A fully homogeneous cluster has CV = 0.
        """
        if len(snapshots) < 2:
            return 0.0
        values = [s.mips_capacity for s in snapshots]
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        cv = math.sqrt(variance) / mean
        # Clamp: CV rarely exceeds 1 for typical cloud heterogeneity
        return min(cv, 1.0)

    def _discretise(self, value: float) -> int:
        """Map a [0, 1] continuous value to a bin index in [0, num_bins-1]."""
        idx = int(value * self.num_bins)
        return min(idx, self.num_bins - 1)

