# Q-Learning Driven Hyper-Heuristic for Energy-Efficient Cloud Workflow Scheduling

A complete Python simulation framework and real-time interactive web dashboard that uses a **Q-learning hyper-heuristic agent** to adaptively select the best low-level scheduling heuristic (HEFT, Min-Min, Max-Min, or Round-Robin) for energy-efficient execution of DAG-based scientific workflows on heterogeneous cloud hosts.

---

## Project Structure

```
cloud_workflow_scheduler/
├── api/
│   ├── __init__.py
│   └── server.py              # FastAPI server (REST + WebSocket live telemetry)
├── frontend/                  # React + Vite + Tailwind CSS + Recharts Dashboard
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ControlPanel.jsx       # Hyperparameter controls & action buttons
│   │   │   ├── MetricCards.jsx        # Real-time analytics cards
│   │   │   ├── ConvergenceChart.jsx   # Reward & Makespan convergence line plot
│   │   │   ├── HeuristicChart.jsx     # Heuristic selection breakdown (Pie/Bar)
│   │   │   ├── HostLoadChart.jsx      # Host CPU load distribution chart
│   │   │   ├── TaskHostMatrix.jsx     # Live task execution & host grid
│   │   │   └── TelemetryConsole.jsx   # Terminal logs & Q-Table inspector
│   │   ├── hooks/
│   │   │   └── useWebSocket.js        # Reconnecting WebSocket hook
│   │   ├── api.js                     # Backend API client
│   │   ├── App.jsx                    # Main unified dashboard view
│   │   ├── index.css                  # Tailwind styles & theme
│   │   └── main.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── parsers/
│   └── dag_parser.py          # Workflow DAG loader (JSON / YAML)
├── core/
│   └── state_evaluator.py     # Real-time cloud state monitor
├── agents/
│   └── ql_agent.py            # Q-learning hyper-heuristic agent
├── heuristics/
│   ├── heft.py                # HEFT scheduling heuristic
│   └── greedy.py              # Min-Min / Max-Min / Round-Robin heuristics
├── simulation/
│   └── engine.py              # Event-driven simulation + energy model
├── utils/
│   └── metrics.py             # Performance metrics & CSV reporting
├── data/
│   ├── montage.json           # Sample Montage workflow (JSON)
│   └── cybershake.yaml        # Sample CyberShake workflow (YAML)
├── main.py                    # Headless CLI simulation entry point
├── run_app.py                 # Concurrent launcher for Backend + Frontend
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Minimum Version |
|-------------|----------------|
| Python      | 3.10+          |
| Node.js     | 18+ (Node 20+ recommended) |
| npm         | 9+             |

---

## Installation

```bash
# Clone or navigate to the project directory
cd cloud_workflow_scheduler

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Frontend dependencies
cd frontend
npm install
cd ..
```

---

## Running the Web Dashboard (Recommended)

To launch both the FastAPI backend server and the React frontend concurrently:

```bash
python run_app.py
```

Once running, access the interfaces:
* **Interactive Web Dashboard**: [http://localhost:5173](http://localhost:5173)
* **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **WebSocket Telemetry Stream**: `ws://localhost:8000/ws/live-stream`

### Dashboard Features
1. **Control Panel**:
   * Interactive sliders for Epsilon ($\varepsilon$), Learning Rate ($\alpha$), Discount Factor ($\gamma$), and Epsilon Decay.
   * Dataset selector: **Synthetic DAG**, **Montage**, or **CyberShake**.
   * Run buttons: **Run 500 Episodes**, **Step &times;1**, **Run 50 Episodes**, and **Reset**.
2. **Real-Time Analytics Cards**:
   * Total Energy (kWh), Makespan (s), Average Host Utilisation (%), and Load Imbalance (LI) index.
3. **Dynamic Recharts Data Visualisations**:
   * **Reward & Makespan Convergence**: Dual-axis line plot across training episodes.
   * **Heuristic Selection Breakdown**: Interactive toggle between Donut Pie and Bar charts for HEFT, Min-Min, Max-Min, and RoundRobin.
   * **Host CPU Load Distribution**: Horizontal bar chart indicating load status per cloud host.
4. **Task DAG & Host Matrix**:
   * Real-time grid displaying task-to-host allocations with start/finish timestamps and energy draw.
5. **Telemetry Console & Q-Table Inspector**:
   * Terminal-style live stream of TD-errors, selected heuristics, and state vectors.
   * Tabbed Q-Table inspector showing action values $Q(s, a)$ and the greedy best policy action.

---

## Headless CLI Simulation

You can also run experiments directly via the CLI without the web interface:

### 1 — Synthetic workflow (default, zero configuration)
```bash
python main.py
```

### 2 — Sample Montage workflow
```bash
python main.py --workflow data/montage.json --episodes 300
```

### 3 — Sample CyberShake workflow (YAML)
```bash
python main.py --workflow data/cybershake.yaml --episodes 150 --hosts 4
```

### 4 — Full custom experiment
```bash
python main.py \
  --tasks 25 \
  --hosts 8 \
  --episodes 500 \
  --alpha 0.15 \
  --gamma 0.95 \
  --epsilon 1.0 \
  --epsilon-end 0.02 \
  --epsilon-decay 0.97 \
  --seed 0 \
  --w-energy 0.5 \
  --w-makespan 0.3 \
  --w-load 0.2 \
  --output results/ \
  --experiment-name montage_exp \
  --verbose
```

---

## API & WebSocket Endpoints

| Method / Protocol | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Returns system status, active host utilisation, and last episode metrics |
| `POST` | `/api/simulate` | Triggers $N$ simulation episodes with custom hyperparameters in background |
| `POST` | `/api/reset` | Resets agent Q-table, host fleet state, and metric histories |
| `GET` | `/api/results` | Returns convergence histories, Q-table stats, and per-heuristic averages |
| `WS` | `/ws/live-stream` | Real-time bi-directional streaming of episode ticks, state vectors, and allocations |

---

## Workflow File Format

### JSON
```json
{
  "workflow": {
    "name": "MyWorkflow",
    "tasks": [
      {
        "id": "t0",
        "cpu_mips": 2000,
        "ram_mb": 512,
        "file_size_mb": 10.0,
        "runtime_s": 3.0
      }
    ],
    "edges": [
      { "src": "t0", "dst": "t1", "data_mb": 5.0 }
    ]
  }
}
```

### YAML
```yaml
workflow:
  name: MyWorkflow
  tasks:
    - id: t0
      cpu_mips: 2000
      ram_mb: 512
      file_size_mb: 10.0
      runtime_s: 3.0
  edges:
    - src: t0
      dst: t1
      data_mb: 5.0
```

---

## Architecture Overview

```
 ┌─────────────────────────────────────────────────────────────────┐
 │               run_app.py (Concurrent Orchestrator)              │
 └──────────────┬───────────────────────────────────┬──────────────┘
                │                                   │
 ┌──────────────▼─────────────┐       ┌─────────────▼──────────────┐
 │    React + Vite Frontend   │◀─WS─▶ │       FastAPI Backend      │
 │       (Port 5173)          │──API─▶│         (Port 8000)        │
 └────────────────────────────┘       └─────────────┬──────────────┘
                                                    │
                                      ┌─────────────▼──────────────┐
                                      │   SimulationEngine         │
                                      │   simulation/engine.py     │
                                      └──────┬──────────────┬──────┘
                                             │              │
                    ┌────────────────────────┴────┐   ┌─────┴──────────────┐
                    │  StateEvaluator & QLAgent   │   │ Low-Level Portfolio│
                    │  (4D State Vector & Q-Table)│   │ (HEFT/MinMin/MaxMin│
                    └─────────────────────────────┘   │  /RoundRobin)      │
                                                      └────────────────────┘
```

---

## Output Files

After any simulation run, output logs and Q-tables are persisted in `results/`:
* `<name>_results.csv`: Per-episode records of energy, makespan, utilisation, reward, and chosen heuristic.
* `<name>_qtable.npz`: Serialised NumPy Q-table for checkpointing or offline analysis.

---

## License

MIT License — free for academic and research use.
