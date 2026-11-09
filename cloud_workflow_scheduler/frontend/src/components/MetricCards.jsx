// src/components/MetricCards.jsx
import { Zap, Clock, Cpu, BarChart2 } from 'lucide-react'

function MetricCard({ icon: Icon, label, value, unit, color, sub }) {
  return (
    <div className="card flex flex-col gap-1">
      <div className={`flex items-center gap-2 ${color}`}>
        <Icon size={15} />
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value mt-1">
        {value}
        <span className="text-sm font-normal text-slate-400 ml-1">{unit}</span>
      </div>
      {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function MetricCards({ metrics, episode, heuristic }) {
  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
      <MetricCard
        icon={Zap} label="Total Energy" color="text-yellow-400"
        value={(metrics.energy_kwh ?? 0).toFixed(5)} unit="kWh"
        sub={`Episode ${episode ?? 0}`}
      />
      <MetricCard
        icon={Clock} label="Makespan" color="text-cyan-400"
        value={(metrics.makespan_s ?? 0).toFixed(2)} unit="s"
        sub={`Heuristic: ${heuristic ?? '—'}`}
      />
      <MetricCard
        icon={Cpu} label="Host Utilisation" color="text-green-400"
        value={(metrics.utilisation_pct ?? 0).toFixed(1)} unit="%"
        sub="Avg across hosts"
      />
      <MetricCard
        icon={BarChart2} label="Load Imbalance" color="text-red-400"
        value={(metrics.load_imbalance ?? 0).toFixed(4)} unit="LI"
        sub="CV of utilisations"
      />
    </div>
  )
}

