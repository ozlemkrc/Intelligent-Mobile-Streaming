import type { SidebarParams, AppState } from '../types'
import { ProgressBar } from './ProgressBar'

interface SliderProps {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void
}
function Slider({ label, value, min, max, step, onChange }: SliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-indigo-400 font-mono">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full bg-slate-700 cursor-pointer" />
    </div>
  )
}

interface Props {
  params: SidebarParams
  personas: Record<string, { name: string }>
  appState: AppState
  jobProgress?: number
  jobRunning?: boolean
  onParamChange: (key: keyof SidebarParams, value: number | string) => void
  onGenerateData: () => void
  onTrainModels: () => void
  onRunSimulation: () => void
}

export function Sidebar({
  params, personas, appState, jobProgress, jobRunning,
  onParamChange, onGenerateData, onTrainModels, onRunSimulation,
}: Props) {
  const btn = (label: string, onClick: () => void, disabled: boolean) => (
    <button
      onClick={onClick} disabled={disabled || jobRunning}
      className="w-full py-2 px-3 rounded-md text-sm font-medium transition-colors
        bg-indigo-600 hover:bg-indigo-500 text-white
        disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {label}
    </button>
  )

  return (
    <aside className="w-64 shrink-0 bg-slate-900 border-r border-slate-800 h-screen overflow-y-auto flex flex-col">
      <div className="p-4 border-b border-slate-800">
        <h1 className="text-lg font-bold text-indigo-400">📡 IMS Dashboard</h1>
      </div>

      <div className="p-4 space-y-5 flex-1">
        {/* Section 1 */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">1 · Persona Data</h2>
          <div className="space-y-1">
            <div className="text-xs text-slate-400">Active persona</div>
            <select
              value={params.activePersona}
              onChange={e => onParamChange('activePersona', e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {Object.entries(personas).map(([k, p]) => (
                <option key={k} value={k}>{p.name}</option>
              ))}
            </select>
          </div>
          <Slider label="Sessions per user" value={params.nSessions} min={5} max={40} step={5} onChange={v => onParamChange('nSessions', v)} />
          <Slider label="Session duration (s)" value={params.sessionDuration} min={120} max={600} step={60} onChange={v => onParamChange('sessionDuration', v)} />
          <div className="space-y-1">
            <div className="text-xs text-slate-400">Data seed</div>
            <input type="number" value={params.dataSeed} min={0} max={9999}
              onChange={e => onParamChange('dataSeed', Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" />
          </div>
          {btn('Generate All Persona Data', onGenerateData, false)}
        </section>

        <div className="border-t border-slate-800" />

        {/* Section 2 */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">2 · Train Models</h2>
          <Slider label="LSTM window (steps)" value={params.lstmWindow} min={5} max={30} step={5} onChange={v => onParamChange('lstmWindow', v)} />
          <Slider label="LSTM epochs" value={params.lstmEpochs} min={10} max={60} step={5} onChange={v => onParamChange('lstmEpochs', v)} />
          <Slider label="Early-stop patience" value={params.lstmPatience} min={3} max={15} step={1} onChange={v => onParamChange('lstmPatience', v)} />
          <Slider label="KNN — k" value={params.knnK} min={1} max={21} step={2} onChange={v => onParamChange('knnK', v)} />
          <Slider label="RF — trees" value={params.rfTrees} min={50} max={300} step={50} onChange={v => onParamChange('rfTrees', v)} />
          {btn('Train on Selected Persona', onTrainModels, !appState.has_data)}
        </section>

        <div className="border-t border-slate-800" />

        {/* Section 3 */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">3 · Streaming</h2>
          <Slider label="Stream starts at hour" value={params.simHour} min={0} max={23} step={0.5} onChange={v => onParamChange('simHour', v)} />
          <Slider label="Duration (s)" value={params.simDuration} min={60} max={600} step={30} onChange={v => onParamChange('simDuration', v)} />
          <div className="space-y-1">
            <div className="text-xs text-slate-400">Sim seed</div>
            <input type="number" value={params.simSeed} min={0} max={9999}
              onChange={e => onParamChange('simSeed', Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" />
          </div>
          {btn('Run Simulation', onRunSimulation, !appState.has_models)}
        </section>

        {/* Job progress */}
        {jobRunning && jobProgress !== undefined && (
          <div className="pt-2">
            <ProgressBar progress={jobProgress} label="Running…" />
          </div>
        )}
      </div>
    </aside>
  )
}
