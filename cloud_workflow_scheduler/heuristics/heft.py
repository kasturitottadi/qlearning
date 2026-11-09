"""
heuristics/heft.py
==================
HEFT (Heterogeneous Earliest Finish Time) scheduling heuristic.

HEFT is a static, list-based scheduling algorithm for DAGs on heterogeneous
processors.  It has two phases:

1. **Ranking phase** – Compute an *upward rank* for each task (sum of average
   computation cost + maximum communication cost along the critical path to
   the exit task).  Tasks are sorted in decreasing rank order.

2. **Selection phase** – Assign each task (in rank order) to the host that
   yields the *earliest finish time*, accounting for insertion into existing
   schedule gaps.

References
----------
Topcuoglu, H., Hariri, S., & Wu, M.-Y. (2002). Performance-effective and
low-complexity task scheduling for heterogeneous computing. *IEEE Transactions
on Parallel and Distributed Systems*, 13(3), 260–274.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models (local aliases to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass
class HEFTTask:
    task_id: str
    runtime_s: float          # Base runtime on a *reference* 1 MIPS host
    cpu_mips: float           # Task MIPS demand
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    upward_rank: float = 0.0
    assigned_host: Optional[str] = None
    start_time: float = 0.0
    finish_time: float = 0.0


@dataclass
class HEFTHost:
    host_id: str
    mips: float               # Host processing capacity in MIPS
    # Ordered list of (start, end) busy intervals
    schedule: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def earliest_available(self) -> float:
        """Time at which this host becomes fully free."""
        if not self.schedule:
            return 0.0
        return max(end for _, end in self.schedule)


# ---------------------------------------------------------------------------
# HEFT implementation
# ---------------------------------------------------------------------------

class HEFT:
    """
    HEFT scheduler.

    Parameters
    ----------
    tasks : dict[str, HEFTTask]
        All tasks keyed by task_id.
    hosts : list[HEFTHost]
        Available cloud hosts.
    avg_bandwidth_mbps : float
        Average inter-host bandwidth for communication cost estimation.
    """

    def __init__(
        self,
        tasks: Dict[str, HEFTTask],
        hosts: List[HEFTHost],
        avg_bandwidth_mbps: float = 100.0,
    ) -> None:
        self.tasks = tasks
        self.hosts = hosts
        self.avg_bw = avg_bandwidth_mbps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self) -> Dict[str, str]:
        """
        Run HEFT and return a task → host assignment mapping.

        Returns
        -------
        dict[str, str]
            Maps each task_id to the assigned host_id.
        """
        self._compute_upward_ranks()
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.upward_rank,
            reverse=True,
        )

        host_map: Dict[str, str] = {}
        for task in sorted_tasks:
            best_host, eft = self._find_best_host(task)
            task.assigned_host = best_host.host_id
            task.finish_time = eft
            # Approximate start time
            task.start_time = eft - self._compute_time(task, best_host)
            best_host.schedule.append((task.start_time, task.finish_time))
            host_map[task.task_id] = best_host.host_id
            logger.debug(
                "HEFT: task=%s → host=%s  [%.2f, %.2f]",
                task.task_id,
                best_host.host_id,
                task.start_time,
                task.finish_time,
            )

        return host_map

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_upward_ranks(self) -> None:
        """Recursively compute upward ranks (memoised)."""
        memo: Dict[str, float] = {}

        def rank(task_id: str) -> float:
            if task_id in memo:
                return memo[task_id]
            task = self.tasks[task_id]
            avg_comp = self._avg_compute_time(task)
            if not task.successors:
                memo[task_id] = avg_comp
                return avg_comp
            max_succ = max(
                self._comm_cost(task_id, s) + rank(s) for s in task.successors
            )
            memo[task_id] = avg_comp + max_succ
            return memo[task_id]

        for tid in self.tasks:
            self.tasks[tid].upward_rank = rank(tid)

    def _avg_compute_time(self, task: HEFTTask) -> float:
        """Average computation time across all hosts."""
        times = [self._compute_time(task, h) for h in self.hosts]
        return sum(times) / len(times) if times else task.runtime_s

    def _compute_time(self, task: HEFTTask, host: HEFTHost) -> float:
        """Actual compute time for `task` on `host` (MIPS-based)."""
        return task.runtime_s * (task.cpu_mips / max(host.mips, 1.0))

    def _comm_cost(self, src_id: str, dst_id: str) -> float:
        """Estimated communication cost between two tasks (seconds)."""
        # Simplified: use average task file size / bandwidth
        # In a real implementation this would use edge data_mb
        src_task = self.tasks[src_id]
        avg_data_mb = 5.0  # placeholder; engine sets real values
        return avg_data_mb / max(self.avg_bw / 8.0, 1e-6)  # MB / (MB/s)

    def _find_best_host(self, task: HEFTTask) -> Tuple[HEFTHost, float]:
        """Return (best_host, earliest_finish_time) for a task."""
        best_host: Optional[HEFTHost] = None
        best_eft = float("inf")

        for host in self.hosts:
            est = self._earliest_start_time(task, host)
            eft = est + self._compute_time(task, host)
            if eft < best_eft:
                best_eft = eft
                best_host = host

        assert best_host is not None
        return best_host, best_eft

    def _earliest_start_time(self, task: HEFTTask, host: HEFTHost) -> float:
        """
        Compute the earliest time task can start on host, considering:
        * predecessor finish times + communication delays
        * host availability
        """
        # Data-ready time: max over all predecessors
        data_ready = 0.0
        for pred_id in task.predecessors:
            pred = self.tasks[pred_id]
            if pred.assigned_host and pred.assigned_host != host.host_id:
                comm = self._comm_cost(pred_id, task.task_id)
            else:
                comm = 0.0
            data_ready = max(data_ready, pred.finish_time + comm)

        return max(data_ready, host.earliest_available)

