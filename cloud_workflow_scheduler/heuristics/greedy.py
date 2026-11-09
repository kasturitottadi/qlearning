"""
heuristics/greedy.py
====================
Greedy low-level heuristics for cloud task scheduling.

Implements three classic greedy strategies:

* **Min-Min** – For each ready task, assign it to the host that gives the
  *minimum completion time*.  Tasks with the minimum completion time across
  all ready tasks are scheduled first.

* **Max-Min** – Like Min-Min but tasks with the *maximum* (over all ready
  tasks) of their minimum completion time are scheduled first.  Favours
  longer tasks receiving best resources.

* **Round-Robin** – Cyclic assignment across hosts regardless of load.
  Simple and fair but workload-agnostic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight host / task interfaces (duck-typed, no circular imports)
# ---------------------------------------------------------------------------

class GreedyHost:
    """Minimal host abstraction used by greedy heuristics."""

    def __init__(self, host_id: str, mips: float) -> None:
        self.host_id = host_id
        self.mips = mips
        self.available_time: float = 0.0  # When host next becomes free


class GreedyTask:
    """Minimal task abstraction used by greedy heuristics."""

    def __init__(
        self,
        task_id: str,
        cpu_mips: float,
        runtime_s: float,
        predecessors: Optional[List[str]] = None,
    ) -> None:
        self.task_id = task_id
        self.cpu_mips = cpu_mips
        self.runtime_s = runtime_s
        self.predecessors: List[str] = predecessors or []
        self.assigned_host: Optional[str] = None
        self.start_time: float = 0.0
        self.finish_time: float = 0.0


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _completion_time(task: GreedyTask, host: GreedyHost) -> float:
    """
    Estimated completion time for *task* on *host*.

    Completion time = max(host_available_time, task_data_ready_time)
                      + task_runtime / host_speed_ratio
    """
    execution_time = task.runtime_s * (task.cpu_mips / max(host.mips, 1e-9))
    return host.available_time + execution_time


def _build_completion_matrix(
    tasks: List[GreedyTask],
    hosts: List[GreedyHost],
) -> Dict[str, Dict[str, float]]:
    """Return a task→host→completion_time matrix."""
    return {
        task.task_id: {host.host_id: _completion_time(task, host) for host in hosts}
        for task in tasks
    }


# ---------------------------------------------------------------------------
# Min-Min
# ---------------------------------------------------------------------------

class MinMin:
    """
    Min-Min heuristic scheduler.

    Iteratively picks the ready task with the smallest *minimum completion time*
    across all hosts and assigns it to its best host.

    Parameters
    ----------
    hosts : list[GreedyHost]
        Available cloud hosts.
    """

    def __init__(self, hosts: List[GreedyHost]) -> None:
        self.hosts = hosts

    def schedule(self, ready_tasks: List[GreedyTask]) -> Dict[str, str]:
        """
        Schedule all ready tasks using Min-Min.

        Parameters
        ----------
        ready_tasks : list[GreedyTask]
            Tasks ready for scheduling (all predecessors finished).

        Returns
        -------
        dict[str, str]
            Maps task_id → host_id.
        """
        remaining = list(ready_tasks)
        assignment: Dict[str, str] = {}

        while remaining:
            matrix = _build_completion_matrix(remaining, self.hosts)

            # For each task find its minimum CT and best host
            task_min: Dict[str, Tuple[str, float]] = {}
            for task in remaining:
                best_host_id = min(matrix[task.task_id], key=matrix[task.task_id].get)  # type: ignore[arg-type]
                task_min[task.task_id] = (best_host_id, matrix[task.task_id][best_host_id])

            # Select the task with the global minimum CT
            chosen_id = min(task_min, key=lambda tid: task_min[tid][1])
            best_host_id, ct = task_min[chosen_id]

            # Apply assignment
            chosen_task = next(t for t in remaining if t.task_id == chosen_id)
            chosen_task.assigned_host = best_host_id
            chosen_task.finish_time = ct

            best_host = next(h for h in self.hosts if h.host_id == best_host_id)
            best_host.available_time = ct

            assignment[chosen_id] = best_host_id
            remaining.remove(chosen_task)

            logger.debug("Min-Min: task=%s → host=%s  CT=%.2f", chosen_id, best_host_id, ct)

        return assignment


# ---------------------------------------------------------------------------
# Max-Min
# ---------------------------------------------------------------------------

class MaxMin:
    """
    Max-Min heuristic scheduler.

    Like Min-Min but selects the task whose *minimum completion time* is
    maximum among all ready tasks.  This tends to schedule long tasks on
    their best hosts early, reducing overall makespan.

    Parameters
    ----------
    hosts : list[GreedyHost]
        Available cloud hosts.
    """

    def __init__(self, hosts: List[GreedyHost]) -> None:
        self.hosts = hosts

    def schedule(self, ready_tasks: List[GreedyTask]) -> Dict[str, str]:
        """
        Schedule all ready tasks using Max-Min.

        Parameters
        ----------
        ready_tasks : list[GreedyTask]
            Tasks ready for scheduling.

        Returns
        -------
        dict[str, str]
            Maps task_id → host_id.
        """
        remaining = list(ready_tasks)
        assignment: Dict[str, str] = {}

        while remaining:
            matrix = _build_completion_matrix(remaining, self.hosts)

            task_min: Dict[str, Tuple[str, float]] = {}
            for task in remaining:
                best_host_id = min(matrix[task.task_id], key=matrix[task.task_id].get)  # type: ignore[arg-type]
                task_min[task.task_id] = (best_host_id, matrix[task.task_id][best_host_id])

            # Select the task with the MAXIMUM of minimum CTs
            chosen_id = max(task_min, key=lambda tid: task_min[tid][1])
            best_host_id, ct = task_min[chosen_id]

            chosen_task = next(t for t in remaining if t.task_id == chosen_id)
            chosen_task.assigned_host = best_host_id
            chosen_task.finish_time = ct

            best_host = next(h for h in self.hosts if h.host_id == best_host_id)
            best_host.available_time = ct

            assignment[chosen_id] = best_host_id
            remaining.remove(chosen_task)

            logger.debug("Max-Min: task=%s → host=%s  CT=%.2f", chosen_id, best_host_id, ct)

        return assignment


# ---------------------------------------------------------------------------
# Round-Robin
# ---------------------------------------------------------------------------

class RoundRobin:
    """
    Round-Robin cyclic task-to-host assignment.

    Parameters
    ----------
    hosts : list[GreedyHost]
        Available cloud hosts.
    """

    def __init__(self, hosts: List[GreedyHost]) -> None:
        self.hosts = hosts
        self._counter: int = 0

    def schedule(self, ready_tasks: List[GreedyTask]) -> Dict[str, str]:
        """
        Assign tasks cyclically across hosts.

        Parameters
        ----------
        ready_tasks : list[GreedyTask]
            Tasks ready for scheduling.

        Returns
        -------
        dict[str, str]
            Maps task_id → host_id.
        """
        assignment: Dict[str, str] = {}
        num_hosts = len(self.hosts)

        for task in ready_tasks:
            host = self.hosts[self._counter % num_hosts]
            self._counter += 1

            execution_time = task.runtime_s * (task.cpu_mips / max(host.mips, 1e-9))
            task.assigned_host = host.host_id
            task.start_time = host.available_time
            task.finish_time = host.available_time + execution_time
            host.available_time = task.finish_time

            assignment[task.task_id] = host.host_id
            logger.debug(
                "RoundRobin: task=%s → host=%s  CT=%.2f",
                task.task_id,
                host.host_id,
                task.finish_time,
            )

        return assignment

