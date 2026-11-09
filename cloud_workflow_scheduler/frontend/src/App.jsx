// src/App.jsx
import { useState, useEffect, useCallback } from 'react'
import { Activity, Wifi, WifiOff, Cloud, RefreshCw } from 'lucide-react'
import { useWebSocket } from './hooks/useWebSocket'
import { getStatus, getResults, postSimulate, postReset } from './api'

import ControlPanel from './components/ControlPanel'
import MetricCards from './components/MetricCards'
import ConvergenceChart from './components/ConvergenceChart'
import HeuristicChart from './components/HeuristicChart'
import HostLoadChart from './components/HostLoadChart'
import TaskHostMatrix from './components/TaskHostMatrix'
import TelemetryConsole from './components/TelemetryConsole'

export default function App() {
  const [params, setParams] = useState({
    workflow: 'synthetic',
    epsilon: 1.0,
    alpha: 0.1,
    gamma: 0.9,
    epsilon_decay: 0.98,
    epsilon_end: 0.05,
    num_tasks: 15,
    num_hosts: 5,
    w_energy: 0.4,
    w_makespan: 0.4,
    w_load: 0.2,
    seed: 42,
  })

  const [running, setRunning] = useState(false)
  const [episode, setEpisode] = useState(0)
  const [lastHeuristic, setLastHeuristic] = useState('—')
  const [metrics, setMetrics] = useState({
    energy_kwh: 0,
    makespan_s: 0,
    utilisation_pct: 0,
    load_imbalance: 0,
    reward: 0,
  })

  const [rewards, setRewards] = useState([])
  const [makespans, setMakespans] = useState([])
  const [heuristicCounts, setHeuristicCounts] = useState({
    HEFT: 0,
    'Min-Min': 0,
    'Max-Min': 0,
    RoundRobin: 0,
  })
  const [hostUtilisation, setHostUtilisation] = useState({})
  const [taskAllocations, setTaskAllocations] = useState([])
  const [qValues, setQValues] = useState({})
  const [stateVector, setStateVector] = useState(null)
  const [tdError, setTdError] = useState(0)
  const [epsilon, setEpsilon] = useState(1.0)
  const [logs, setLogs] = useState([])

  const addLog = useCallback((message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs((prev) => [...prev.slice(-150), { time, message, type }])
  }, [])

  // Handle live WebSocket incoming events
  const handleWebSocketMessage = useCallback(
    (msg) => {
      if (!msg) return

      if (msg.type === 'init') {
        if (msg.episode) setEpisode(msg.episode)
        if (msg.rewards) setRewards(msg.rewards)
        if (msg.makespans) setMakespans(msg.makespans)
        if (msg.heuristic_counts) setHeuristicCounts(msg.heuristic_counts)
        if (msg.host_utilisation) setHostUtilisation(msg.host_utilisation)
        addLog('Connected to Live Stream Engine.', 'success')
      } else if (msg.type === 'episode_tick') {
        setEpisode(msg.episode)
        setLastHeuristic(msg.heuristic)
        setMetrics(msg.metrics)
        setRewards((prev) => [...prev.slice(-499), msg.metrics.reward])
        setMakespans((prev) => [...prev.slice(-499), msg.metrics.makespan_s])
        setHeuristicCounts(msg.heuristic_counts)
        setHostUtilisation(msg.host_utilisation)
        setTaskAllocations(msg.task_allocations || [])
        setQValues(msg.q_values || {})
        setStateVector(msg.state_vector)
        setTdError(msg.td_error)
        setEpsilon(msg.epsilon)

        addLog(
          `Ep #${msg.episode} [${msg.heuristic}] R: ${msg.metrics.reward.toFixed(
            4
          )} | E: ${msg.metrics.energy_kwh.toFixed(5)} kWh | MS: ${msg.metrics.makespan_s.toFixed(
            2
          )}s | TD-Err: ${msg.td_error.toFixed(4)}`,
          'action'
        )
      } else if (msg.type === 'simulation_complete') {
        setRunning(false)
        addLog(
          `Batch complete! Total episodes: ${msg.total_episodes}. Mean Reward (last 10): ${msg.convergence?.mean_reward_last_10}`,
          'success'
        )
      } else if (msg.type === 'reset') {
        setEpisode(0)
        setRewards([])
        setMakespans([])
        setHeuristicCounts({ HEFT: 0, 'Min-Min': 0, 'Max-Min': 0, RoundRobin: 0 })
        setMetrics({ energy_kwh: 0, makespan_s: 0, utilisation_pct: 0, load_imbalance: 0, reward: 0 })
        setTaskAllocations([])
        setQValues({})
        setStateVector(null)
        setTdError(0)
        setRunning(false)
        addLog('System state & Q-table reset to defaults.', 'warning')
      }
    },
    [addLog]
  )

  const { connected } = useWebSocket(handleWebSocketMessage)

  // Fetch initial system status on mount
  useEffect(() => {
    async function loadInitial() {
      try {
        const status = await getStatus()
        setRunning(status.running)
        setEpisode(status.episode)
        setEpsilon(status.epsilon)
        if (status.last_metrics) {
          setMetrics(status.last_metrics)
          setLastHeuristic(status.last_metrics.heuristic || '—')
        }
        if (status.host_utilisation) setHostUtilisation(status.host_utilisation)
        if (status.heuristic_counts) setHeuristicCounts(status.heuristic_counts)

        const results = await getResults()
        if (results.rewards) setRewards(results.rewards)
        if (results.makespans) setMakespans(results.makespans)
      } catch (err) {
        addLog(`Backend sync error: ${err.message}`, 'error')
      }
    }
    loadInitial()
  }, [addLog])

  const handleRun = async (count = 50) => {
    setRunning(true)
    addLog(`Initiating simulation run: ${count} episodes [${params.workflow}]...`, 'update')
    try {
      await postSimulate({
        ...params,
        episodes: count,
      })
    } catch (err) {
      setRunning(false)
      addLog(`Failed to start simulation: ${err.message}`, 'error')
    }
  }

  const handleReset = async () => {
    try {
      await postReset()
    } catch (err) {
      addLog(`Reset failed: ${err.message}`, 'error')
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <Cloud size={20} />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white flex items-center gap-2">
              Q-Learning Hyper-Heuristic Workflow Scheduler
              <span className="text-[10px] font-mono font-normal px-2 py-0.5 rounded bg-indigo-950 border border-indigo-700/50 text-indigo-300">
                v1.0-live
              </span>
            </h1>
            <p className="text-[11px] text-slate-400 font-mono">
              Energy-Efficient DAG Execution on Heterogeneous Cloud
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700">
            <Activity size={13} className={running ? 'text-amber-400 animate-spin' : 'text-slate-500'} />
            <span className={running ? 'text-amber-300' : 'text-slate-400'}>
              {running ? 'Simulating...' : 'Idle'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700">
            {connected ? (
              <>
                <Wifi size={13} className="text-emerald-400" />
                <span className="text-emerald-400">Live WS</span>
              </>
            ) : (
              <>
                <WifiOff size={13} className="text-rose-400" />
                <span className="text-rose-400">WS Disconnected</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-4 gap-6 max-w-[1700px] w-full mx-auto">
        {/* Left Column: Control Panel */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <ControlPanel
            params={params}
            setParams={setParams}
            onRun={handleRun}
            onStep={() => handleRun(1)}
            onReset={handleReset}
            running={running}
          />
        </div>

        {/* Right Columns: Analytics, Charts, Matrices & Telemetry */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          {/* Top Real-Time Metrics Cards */}
          <MetricCards
            metrics={metrics}
            episode={episode}
            heuristic={lastHeuristic}
          />

          {/* Dynamic Recharts Row */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2">
              <ConvergenceChart rewards={rewards} makespans={makespans} />
            </div>
            <div>
              <HeuristicChart counts={heuristicCounts} />
            </div>
          </div>

          {/* Host Load & Live Allocations */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <div>
              <HostLoadChart hostUtilisation={hostUtilisation} />
            </div>
            <div className="xl:col-span-2">
              <TaskHostMatrix
                taskAllocations={taskAllocations}
                hosts={Object.keys(hostUtilisation)}
              />
            </div>
          </div>

          {/* Telemetry Console & Q-Table Inspector */}
          <TelemetryConsole
            logs={logs}
            qValues={qValues}
            stateVector={stateVector}
            tdError={tdError}
            epsilon={epsilon}
            onClear={() => setLogs([])}
          />
        </div>
      </main>
    </div>
  )
}

