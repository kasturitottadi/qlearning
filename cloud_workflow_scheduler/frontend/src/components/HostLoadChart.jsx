// src/components/HostLoadChart.jsx
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-300 font-semibold">{payload[0].payload.host}</p>
      <p className="text-green-400">Utilisation: {payload[0].value.toFixed(1)}%</p>
    </div>
  )
}

function getBarColor(util) {
  if (util > 80) return '#ef4444' // red - high load
  if (util > 50) return '#eab308' // yellow - medium
  if (util > 0) return '#22c55e'  // green - normal
  return '#475569'                // slate - idle
}

export default function HostLoadChart({ hostUtilisation }) {
  const data = Object.entries(hostUtilisation || {}).map(([host, util]) => ({
    host,
    utilisation: Number(util) || 0,
  }))

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
          Host CPU Load Distribution
        </h3>
        <span className="text-[10px] text-slate-500 font-mono">
          {data.length} active hosts
        </span>
      </div>

      {data.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-slate-600 text-sm">
          No host telemetry yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 4, right: 20, left: 10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fill: '#64748b', fontSize: 10 }}
              unit="%"
            />
            <YAxis
              type="category"
              dataKey="host"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              width={60}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="utilisation" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.utilisation)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

