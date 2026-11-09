// src/components/ControlPanel.jsx
import { Play, StepForward, RotateCcw, Zap } from 'lucide-react'

const WORKFLOWS = [
  { value: 'synthetic', label: 'Synthetic DAG' },
  { value: 'montage',   label: 'Montage' },
  { value: 'cybershake', label: 'CyberShake' },
]

function Slider({ label, min, max, step, value, onChange }) {
  return (
    <div className="mb-4">
      <div className="slider-label">
        <span>{label}</span>
        <span className="text-indigo-400 font-semibold">{value}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export default function ControlPanel({ params, setParams, onRun, onStep, onReset, running }) {
  const set = (key) => (val) => setParams(p => ({ ...p, [key]: val }))

  return (
    <div className="card flex flex-col gap-4">
      <h2 className="text-sm font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
        <Zap size={14} className="text-indigo-400" /> Control Panel
      </h2>

      {/* Workflow selector */}
      <div>
        <div className="slider-label"><span>Workflow Dataset</span></div>
        <select value={params.workflow} onChange={e => set('workflow')(e.target.value)}>
          {WORKFLOWS.map(w => <option key={w.value} value={w.value}>{w.label}</option>)}
        </select>
      </div>

      <Slider label="Epsilon (ε) – Exploration" min={0} max={1} step={0.01}
        value={params.epsilon} onChange={set('epsilon')} />
      <Slider label="Learning Rate (α)" min={0.001} max={1} step={0.001}
        value={params.alpha} onChange={set('alpha')} />
      <Slider label="Discount Factor (γ)" min={0} max={1} step={0.01}
        value={params.gamma} onChange={set('gamma')} />
      <Slider label="Epsilon Decay" min={0.9} max={1} step={0.001}
        value={params.epsilon_decay} onChange={set('epsilon_decay')} />

      {/* Task / Host counts */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="slider-label"><span>Tasks</span><span className="text-indigo-400">{params.num_tasks}</span></div>
          <input type="range" min={5} max={40} step={1}
            value={params.num_tasks} onChange={e => set('num_tasks')(Number(e.target.value))} />
        </div>
        <div>
          <div className="slider-label"><span>Hosts</span><span className="text-indigo-400">{params.num_hosts}</span></div>
          <input type="range" min={2} max={10} step={1}
            value={params.num_hosts} onChange={e => set('num_hosts')(Number(e.target.value))} />
        </div>
      </div>

      {/* Reward weights */}
      <div className="border-t border-slate-700 pt-3">
        <p className="text-xs text-slate-500 mb-2">Reward Weights (must sum ≈ 1)</p>
        <Slider label="w_energy" min={0} max={1} step={0.05}
          value={params.w_energy} onChange={set('w_energy')} />
        <Slider label="w_makespan" min={0} max={1} step={0.05}
          value={params.w_makespan} onChange={set('w_makespan')} />
        <Slider label="w_load" min={0} max={1} step={0.05}
          value={params.w_load} onChange={set('w_load')} />
      </div>

      {/* Action buttons */}
      <div className="flex flex-col gap-2 mt-2">
        <button className="btn-primary flex items-center justify-center gap-2"
          onClick={() => onRun(500)} disabled={running}>
          <Play size={14} /> Run 500 Episodes
        </button>
        <div className="grid grid-cols-2 gap-2">
          <button className="btn-secondary flex items-center justify-center gap-2"
            onClick={() => onRun(1)} disabled={running}>
            <StepForward size={14} /> Step ×1
          </button>
          <button className="btn-danger flex items-center justify-center gap-2"
            onClick={onReset} disabled={running}>
            <RotateCcw size={14} /> Reset
          </button>
        </div>
        <button className="btn-secondary flex items-center justify-center gap-2"
          onClick={() => onRun(50)} disabled={running}>
          <Play size={14} /> Run 50 Episodes
        </button>
      </div>
    </div>
  )
}

