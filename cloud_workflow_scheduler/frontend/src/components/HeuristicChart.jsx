// src/components/HeuristicChart.jsx
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import { useState } from 'react'

const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#ef4444']
const HEURISTICS = ['HEFT', 'Min-Min', 'Max-Min', 'RoundRobin']

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p style={{ color: payload[0].fill }}>{payload[0].name}: {payload[0].value}</p>
    </div>
  )
}

export default function HeuristicChart({ counts }) {
  const [mode, setMode] = useState('pie')

  const data = HEURISTICS.map((h, i) => ({
    name: h,
    value: counts?.[h] ?? 0,
    color: COLORS[i],
  })).filter(d => d.value > 0)

  if (!data.length) {
    return (
      <div className="card flex items-center justify-center h-48 text-slate-600 text-sm">
        No episodes yet
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
          Heuristic Selection
        </h3>
        <div className="flex gap-1">
          {['pie', 'bar'].map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`text-xs px-2 py-0.5 rounded ${mode === m ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-400'}`}>
              {m}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        {mode === 'pie' ? (
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={45} outerRadius={75}
              paddingAngle={3} dataKey="value" nameKey="name">
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
          </PieChart>
        ) : (
          <BarChart data={data} margin={{ top: 4, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" name="Episodes" radius={[4,4,0,0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}

