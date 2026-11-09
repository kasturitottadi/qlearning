"""
parsers/dag_parser.py
=====================
Workflow DAG loader and validator.

Supports loading synthetic or standard workflow DAGs from JSON / YAML files.
Each workflow file must conform to the schema described below.

JSON Schema (example)
---------------------
{
  "workflow": {
    "name": "Montage",
    "tasks": [
      {
        "id": "t0",
        "cpu_mips": 1500,
        "ram_mb": 512,
        "file_size_mb": 10.0,
        "runtime_s": 2.5
      }
    ],
    "edges": [
      {"src": "t0", "dst": "t1", "data_mb": 5.0}
    ]
  }
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # pip install pyyaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-class definitions
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a single workflow task (DAG node)."""

    task_id: str
    cpu_mips: float          # Required CPU capacity in MIPS
    ram_mb: float            # Required RAM in MB
    file_size_mb: float      # Input file size in MB
    runtime_s: float         # Base runtime in seconds (on reference host)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)

    # Populated by the scheduler / engine
    assigned_host: Optional[str] = None
    start_time: float = 0.0
    finish_time: float = 0.0


@dataclass
class Edge:
    """Represents a directed dependency edge between two tasks."""

    src: str          # Source task ID
    dst: str          # Destination task ID
    data_mb: float    # Data transferred along this edge in MB


@dataclass
class Workflow:
    """Container for a complete DAG-based workflow."""

    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # Populated after parsing
    entry_tasks: List[str] = field(default_factory=list)   # No predecessors
    exit_tasks: List[str] = field(default_factory=list)    # No successors

    def task_list(self) -> List[Task]:
        """Return tasks ordered topologically (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {tid: 0 for tid in self.tasks}
        for edge in self.edges:
            in_degree[edge.dst] += 1

        queue: List[str] = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered: List[Task] = []

        while queue:
            tid = queue.pop(0)
            ordered.append(self.tasks[tid])
            for suc in self.tasks[tid].successors:
                in_degree[suc] -= 1
                if in_degree[suc] == 0:
                    queue.append(suc)

        if len(ordered) != len(self.tasks):
            raise ValueError("Workflow DAG contains a cycle!")

        return ordered


# ---------------------------------------------------------------------------
# Parser implementation
# ---------------------------------------------------------------------------

class DAGParser:
    """
    Parses workflow DAG files (JSON or YAML) into :class:`Workflow` objects.

    Parameters
    ----------
    filepath : str | Path
        Path to the workflow definition file.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".json", ".yaml", ".yml")

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self._validate_path()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Workflow:
        """
        Load and parse the workflow file.

        Returns
        -------
        Workflow
            A fully constructed :class:`Workflow` instance.

        Raises
        ------
        ValueError
            If the file format is unsupported or the schema is invalid.
        FileNotFoundError
            If the given path does not exist.
        """
        raw = self._load_raw()
        return self._build_workflow(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_path(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Workflow file not found: {self.filepath}")
        if self.filepath.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{self.filepath.suffix}'. "
                f"Expected one of {self.SUPPORTED_EXTENSIONS}."
            )

    def _load_raw(self) -> dict:
        """Read the file and return a raw Python dict."""
        text = self.filepath.read_text(encoding="utf-8")
        if self.filepath.suffix == ".json":
            return json.loads(text)
        return yaml.safe_load(text)

    def _build_workflow(self, raw: dict) -> Workflow:
        """Convert raw dict to a :class:`Workflow` with validated tasks/edges."""
        wf_data = raw.get("workflow", raw)  # accept top-level or nested

        name: str = wf_data.get("name", "unnamed_workflow")
        workflow = Workflow(name=name)

        # --- Tasks -------------------------------------------------------
        raw_tasks: List[dict] = wf_data.get("tasks", [])
        if not raw_tasks:
            raise ValueError("Workflow definition contains no tasks.")

        for t in raw_tasks:
            task_id = str(t["id"])
            task = Task(
                task_id=task_id,
                cpu_mips=float(t.get("cpu_mips", 1000.0)),
                ram_mb=float(t.get("ram_mb", 256.0)),
                file_size_mb=float(t.get("file_size_mb", 1.0)),
                runtime_s=float(t.get("runtime_s", 1.0)),
            )
            workflow.tasks[task_id] = task
            logger.debug("Parsed task %s", task_id)

        # --- Edges -------------------------------------------------------
        raw_edges: List[dict] = wf_data.get("edges", [])
        for e in raw_edges:
            src, dst = str(e["src"]), str(e["dst"])
            if src not in workflow.tasks:
                raise ValueError(f"Edge references unknown source task '{src}'")
            if dst not in workflow.tasks:
                raise ValueError(f"Edge references unknown destination task '{dst}'")

            edge = Edge(src=src, dst=dst, data_mb=float(e.get("data_mb", 0.0)))
            workflow.edges.append(edge)
            workflow.tasks[src].successors.append(dst)
            workflow.tasks[dst].predecessors.append(src)

        # --- Entry / Exit tasks -----------------------------------------
        workflow.entry_tasks = [
            tid for tid, t in workflow.tasks.items() if not t.predecessors
        ]
        workflow.exit_tasks = [
            tid for tid, t in workflow.tasks.items() if not t.successors
        ]

        logger.info(
            "Parsed workflow '%s': %d tasks, %d edges, "
            "%d entry task(s), %d exit task(s).",
            name,
            len(workflow.tasks),
            len(workflow.edges),
            len(workflow.entry_tasks),
            len(workflow.exit_tasks),
        )
        return workflow


# ---------------------------------------------------------------------------
# Synthetic workflow generator (for quick testing)
# ---------------------------------------------------------------------------

def generate_synthetic_workflow(
    num_tasks: int = 10,
    seed: int = 42,
) -> Workflow:
    """
    Generate a random DAG workflow for testing purposes.

    Parameters
    ----------
    num_tasks : int
        Number of tasks in the DAG.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Workflow
        A synthetic :class:`Workflow` instance.
    """
    import random

    rng = random.Random(seed)
    workflow = Workflow(name=f"synthetic_{num_tasks}t")

    # Create tasks
    for i in range(num_tasks):
        tid = f"t{i}"
        workflow.tasks[tid] = Task(
            task_id=tid,
            cpu_mips=rng.uniform(500.0, 3000.0),
            ram_mb=rng.choice([128.0, 256.0, 512.0, 1024.0]),
            file_size_mb=rng.uniform(1.0, 50.0),
            runtime_s=rng.uniform(0.5, 10.0),
        )

    # Create a random DAG (each task may depend on earlier tasks)
    for i in range(1, num_tasks):
        # Connect to 1–2 random predecessors
        num_preds = rng.randint(1, min(2, i))
        preds = rng.sample(range(i), num_preds)
        for p in preds:
            src, dst = f"t{p}", f"t{i}"
            edge = Edge(src=src, dst=dst, data_mb=rng.uniform(1.0, 20.0))
            workflow.edges.append(edge)
            workflow.tasks[src].successors.append(dst)
            workflow.tasks[dst].predecessors.append(src)

    workflow.entry_tasks = [
        tid for tid, t in workflow.tasks.items() if not t.predecessors
    ]
    workflow.exit_tasks = [
        tid for tid, t in workflow.tasks.items() if not t.successors
    ]

    logger.info(
        "Generated synthetic workflow '%s': %d tasks, %d edges.",
        workflow.name,
        len(workflow.tasks),
        len(workflow.edges),
    )
    return workflow

