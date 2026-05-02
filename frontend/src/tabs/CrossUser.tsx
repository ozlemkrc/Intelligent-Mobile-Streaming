import { useState } from 'react'
import { api } from '../api'
import { PlotlyChart } from '../components/PlotlyChart'
import { ProgressBar } from '../components/ProgressBar'
import { useJob } from '../hooks/useJob'
import type { PlotlyFigure } from '../types'

interface Props { hasData: boolean }

interface CrossUserResult {
  table: Record<string, unknown>[]
  mean_delta_acc: number
  personal_wins: number
  total_personas: number
  chart: PlotlyFigure
}

export function CrossUser({ hasData }: Props) {
  const [epochs, setEpochs] = useState(25)
  const [window_, setWindow] = useState(15)
  const [patience, setPatience] = useState(7)
  const [result, setResult] = useState<CrossUserResult | null>(null)

  const job = useJob(() => {
    api.getCrossUserData().then(r => setResult(r as CrossUserResult)).catch(console.error)
  })

  if (!hasData) return <p className="text-slate-400">Generate persona data from the sidebar first.</p>

  const Slider = ({ label, value, min, max, step, onChange }: {
    label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void
  }) => (
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

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Cross-User Personalisation Experiment</h2>
        <p className="text-slate-400 text-sm max-w-2xl">
          For each user: train a <strong className="text-slate-200">personal model</strong> on their data only, and
          a <strong className="text-slate-200">generic model</strong> on all other users' data. Both are evaluated on
          the same held-out test set (last 20%). If the personal model consistently outperforms, personalisation is proven.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4 max-w-lg">
        <Slider label="LSTM epochs" value={epochs} min={10} max={60} step={5} onChange={setEpochs} />
        <Slider label="Window (steps)" value={window_} min={5} max={30} step={5} onChange={setWindow} />
        <Slider label="Patience" value={patience} min={3} max={15} step={1} onChange={setPatience} />
      </div>

      <button
        onClick={async () => {
          const { job_id } = await api.runCrossUser({ epochs, window: window_, patience })
          job.start(job_id)
        }}
        disabled={job.isRunning}
        className="py-2 px-4 rounded-md text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-40"
      >
        Run Cross-User Experiment
      </button>

      {job.isRunning && <ProgressBar progress={job.progress} label="Training models…" />}
      {job.error && <p className="text-red-400 text-sm">{job.error}</p>}

      {result && (
        <div className="space-y-6">
          <section>
            <h3 className="text-base font-medium text-slate-300 mb-3">Results — Personal vs Generic Accuracy</h3>
            <div className="overflow-auto rounded-lg border border-slate-700 mb-3">
              <table className="w-full text-sm">
                <thead className="bg-slate-800">
                  <tr>{Object.keys(result.table[0] ?? {}).map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                  ))}</tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {result.table.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-800/50">
                      {Object.entries(row).map(([k, v], j) => {
                        const isDelta = k.startsWith('Δ')
                        const num = typeof v === 'number' ? v : null
                        return (
                          <td key={j} className={`px-4 py-2 font-mono ${isDelta && num !== null
                            ? num > 0 ? 'text-emerald-400' : num < 0 ? 'text-red-400' : 'text-slate-400'
                            : 'text-slate-200'}`}>
                            {num !== null ? num.toFixed(4) : String(v)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-slate-300">
              <strong className="text-indigo-400">Mean Δ accuracy:</strong>{' '}
              <code className="font-mono">{result.mean_delta_acc >= 0 ? '+' : ''}{result.mean_delta_acc.toFixed(4)}</code>
              {' · '}
              <strong className="text-indigo-400">Personal wins:</strong>{' '}
              <code className="font-mono">{result.personal_wins}/{result.total_personas}</code> personas
            </p>
          </section>

          <PlotlyChart figure={result.chart} className="h-96" />
        </div>
      )}
    </div>
  )
}
