// src/components/TaskHostMatrix.jsx
import { Server, CheckCircle2, Clock } from 'lucide-react'

export default function TaskHostMatrix({ taskAllocations, hosts }) {
  const hostList = hosts && hosts.length > 0 
    ? hosts 
    : Array.from(new Set((taskAllocations || []).map(t => t.host_id)))

  // Group allocations by host
  const allocationsByHost = (taskAllocations || []).reduce((acc, t) => {
    if (!acc[t.host_id]) acc[t.host_id] = []
    acc[t.host_id].push(t)
    return acc
  }, {})

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
          <Server size={14} className="text-indigo-400" />
          Task DAG &amp; Host Allocation Matrix
        </h3>
        <span className="text-[10px] text-slate-500 font-mono">
          Last Episode: {taskAllocations?.length || 0} tasks mapped
        </span>
      </div>

      {hostList.length === 0 ? (
        <div className="flex items-center justify-center h-36 text-slate-600 text-sm">
          Run a simulation step to view host allocations
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 max-h-72 overflow-y-auto pr-1">
          {hostList.map((hostId) => {
            const tasks = allocationsByHost[hostId] || []
            return (
              <div
                key={hostId}
                className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 flex flex-col"
              >
                <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2">
                  <span className="text-xs font-bold text-indigo-300">{hostId}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    {tasks.length} tasks
                  </span>
                </div>

                <div className="flex-1 flex flex-col gap-1.5 overflow-y-auto max-h-48 pr-1">
                  {tasks.length === 0 ? (
                    <span className="text-[11px] text-slate-600 italic">Idle / No tasks</span>
                  ) : (
                    tasks.map((task) => (
                      <div
                        key={task.task_id}
                        className="bg-slate-900 border border-slate-700/50 rounded px-2 py-1 text-[11px] flex flex-col gap-0.5 hover:border-indigo-500/50 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-slate-200">{task.task_id}</span>
                          <CheckCircle2 size={11} className="text-emerald-400" />
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                          <span className="flex items-center gap-1">
                            <Clock size={9} /> {task.start}s &rarr; {task.finish}s
                          </span>
                          <span className="text-amber-400/90">{task.energy_ws} Ws</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

