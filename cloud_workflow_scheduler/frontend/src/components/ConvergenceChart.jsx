// src/components/ConvergenceChart.jsx
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">Episode {label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function ConvergenceChart({ rewards, makespans }) {
  const data = rewards.map((r, i) => ({
    ep: i + 1,
    reward: r,
    makespan: makespans[i] ?? 0,
  }))

  // Downsample if > 300 points for perf
  const display = data.length > 300
    ? data.filter((_, i) => i % Math.ceil(data.length / 300) === 0)
    : data

  return (
    <div className="card">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
        Reward &amp; Makespan Convergence
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={display} margin={{ top: 4, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="ep" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Episode', fill: '#64748b', fontSize: 10, position: 'insideBottomRight', offset: -5 }} />
          <YAxis yAxisId="left" tick={{ fill: '#64748b', fontSize: 10 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 10 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
          <Line yAxisId="left" type="monotone" dataKey="reward" name="Reward"
            stroke="#6366f1" strokeWidth={1.5} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="makespan" name="Makespan (s)"
            stroke="#22d3ee" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

