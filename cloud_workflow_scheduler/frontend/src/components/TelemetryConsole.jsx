// src/components/TelemetryConsole.jsx
import { useRef, useEffect, useState } from 'react'
import { Terminal, Table, Trash2 } from 'lucide-react'

export default function TelemetryConsole({ logs, qValues, stateVector, tdError, epsilon, onClear }) {
  const [activeTab, setActiveTab] = useState('console')
  const terminalEndRef = useRef(null)

  useEffect(() => {
    if (activeTab === 'console') {
      terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, activeTab])

  return (
    <div className="card flex flex-col h-72">
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('console')}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded font-semibold transition-colors ${
              activeTab === 'console'
                ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Terminal size={13} />
            Live Telemetry Console
          </button>
          <button
            onClick={() => setActiveTab('qtable')}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded font-semibold transition-colors ${
              activeTab === 'qtable'
                ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Table size={13} />
            Q-Table Inspector
          </button>
        </div>

        <div className="flex items-center gap-3">
          {activeTab === 'console' && (
            <button
              onClick={onClear}
              title="Clear Console"
              className="text-slate-500 hover:text-slate-300 transition-colors p-1"
            >
              <Trash2 size={13} />
            </button>
          )}
          <span className="text-[10px] text-slate-500 font-mono">
            ε: {epsilon?.toFixed(4) ?? '1.0000'} | TD-Err: {tdError?.toFixed(4) ?? '0.0000'}
          </span>
        </div>
      </div>

      {activeTab === 'console' ? (
        <div className="flex-1 overflow-y-auto font-mono text-[11px] leading-relaxed bg-slate-950 p-2.5 rounded border border-slate-850">
          {logs.length === 0 ? (
            <div className="text-slate-600 italic">Awaiting simulation events...</div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="log-entry py-0.5 text-slate-300 flex items-start gap-2">
                <span className="text-slate-600 select-none">[{log.time}]</span>
                <span className={getLogTypeClass(log.type)}>{log.message}</span>
              </div>
            ))
          )}
          <div ref={terminalEndRef} />
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto text-xs bg-slate-950 p-3 rounded border border-slate-850 flex flex-col gap-3">
          <div>
            <div className="text-slate-400 text-[11px] mb-1">
              Current State Vector:{' '}
              <span className="text-cyan-400 font-mono">
                {stateVector ? `(${stateVector.join(', ')})` : '(—)'}
              </span>
            </div>
            <div className="text-[10px] text-slate-500">
              Format: (Avg CPU Util Bin, Mem Pressure Bin, Queue Length Bin, Host Heterogeneity Bin)
            </div>
          </div>

          <div className="border border-slate-800 rounded overflow-hidden">
            <table className="w-full text-left font-mono text-[11px]">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-1.5 px-3">Action Heuristic</th>
                  <th className="py-1.5 px-3">Q-Value Q(s, a)</th>
                  <th className="py-1.5 px-3">Policy Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850 text-slate-300">
                {qValues && Object.keys(qValues).length > 0 ? (
                  (() => {
                    const maxVal = Math.max(...Object.values(qValues).map(Number))
                    return Object.entries(qValues).map(([heur, val]) => {
                      const isBest = Number(val) === maxVal && maxVal !== 0
                      return (
                        <tr key={heur} className={isBest ? 'bg-indigo-950/40' : ''}>
                          <td className="py-1.5 px-3 font-semibold text-slate-200">{heur}</td>
                          <td className="py-1.5 px-3 text-cyan-400">{Number(val).toFixed(5)}</td>
                          <td className="py-1.5 px-3">
                            {isBest ? (
                              <span className="px-1.5 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-semibold">
                                Best Action (Greedy)
                              </span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  })()
                ) : (
                  <tr>
                    <td colSpan={3} className="py-4 text-center text-slate-600 italic">
                      No active state Q-values recorded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function getLogTypeClass(type) {
  switch (type) {
    case 'success':
      return 'text-emerald-400'
    case 'warning':
      return 'text-amber-400'
    case 'error':
      return 'text-rose-400'
    case 'action':
      return 'text-cyan-300'
    case 'update':
      return 'text-indigo-400'
    default:
      return 'text-slate-300'
  }
}

