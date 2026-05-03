import { useState, useEffect } from 'react'
import { api } from '../api'
import { useJob } from '../hooks/useJob'
import { Section } from '../components/Section'
import { SliderField } from '../components/SliderField'
import { ProgressBar } from '../components/ProgressBar'
import { MetricCard } from '../components/MetricCard'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure, AppState, CrossUserRow } from '../types'

interface Props {
  appState: AppState
  onStateChange: () => void
}

interface CrossUserData {
  table: CrossUserRow[]
  mean_delta_acc: number
  personal_wins: number
  total_personas: number
  chart: PlotlyFigure
}

export function ExperimentPage({ appState, onStateChange }: Props) {
  const [epochs, setEpochs]     = useState(25)
  const [window, setWindow]     = useState(15)
  const [patience, setPatience] = useState(7)
  const [error, setError]       = useState<string | null>(null)
  const [data, setData]         = useState<CrossUserData | null>(null)

  const job = useJob(() => { onStateChange(); loadData() })

  useEffect(() => {
    if (appState.has_crossuser) loadData()
  }, [appState.has_crossuser])

  async function loadData() {
    try {
      const d = await api.getCrossUserData()
      setData(d as CrossUserData)
    } catch { /* not ready */ }
  }

  async function handleRun() {
    setError(null)
    try {
      const { job_id } = await api.runCrossUser({ epochs, window, patience })
      job.start(job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="px-8 py-7 max-w-5xl">

      <Section
        title="Personalization Proof"
        subtitle="For each user: train a personal model on their data, and a generic model on all other users' data. Compare accuracy on the user's held-out test set."
      >
        <div className="flex flex-wrap items-end gap-5">
          <SliderField label="LSTM epochs" value={epochs} min={10} max={60} step={5} onChange={setEpochs} className="w-36" />
          <SliderField label="Window (steps)" value={window} min={5} max={30} step={5} onChange={setWindow} className="w-36" />
          <SliderField label="Patience" value={patience} min={3} max={15} onChange={setPatience} className="w-28" />
          <button
            onClick={handleRun}
            disabled={job.isRunning || !appState.has_data}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded transition-colors"
          >
            {job.isRunning ? 'Running…' : 'Run Experiment'}
          </button>
        </div>
        {job.isRunning && (
          <div className="mt-4 w-80">
            <ProgressBar progress={job.progress} label="Training personal + generic models" />
          </div>
        )}
        {(error || job.error) && <p className="mt-3 text-sm text-red-400">{error ?? job.error}</p>}
        {!appState.has_data && <p className="mt-3 text-sm text-zinc-500">Generate persona data first.</p>}
      </Section>

      {data && (
        <>
          <div className="grid grid-cols-3 gap-3 mb-10">
            <MetricCard
              label="Mean accuracy delta"
              value={`${data.mean_delta_acc >= 0 ? '+' : ''}${(data.mean_delta_acc * 100).toFixed(1)}%`}
              positive={data.mean_delta_acc > 0}
              negative={data.mean_delta_acc < 0}
            />
            <MetricCard
              label="Personal model wins"
              value={`${data.personal_wins} / ${data.total_personas}`}
              positive={data.personal_wins === data.total_personas}
            />
            <MetricCard
              label="Personas tested"
              value={String(data.total_personas)}
            />
          </div>

          <Section title="Results by Persona">
            <div className="border border-zinc-800 rounded-lg overflow-hidden mb-6">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/60">
                    <th className="text-left px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Persona</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Personal Acc</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Generic Acc</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Delta</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Personal F1</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Generic F1</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Delta F1</th>
                  </tr>
                </thead>
                <tbody>
                  {data.table.map((row, i) => {
                    const deltaAcc = row['Delta Acc']
                    const deltaF1  = row['Delta F1']
                    return (
                      <tr key={i} className={`border-b border-zinc-800/40 ${i % 2 === 0 ? 'bg-zinc-900/20' : ''}`}>
                        <td className="px-4 py-3 text-zinc-200 font-medium">{row['Persona']}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-100">{row['Personal Acc'].toFixed(3)}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-400">{row['Generic Acc'].toFixed(3)}</td>
                        <td className={`px-4 py-3 text-right tabular-nums font-medium ${deltaAcc > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {deltaAcc > 0 ? '+' : ''}{deltaAcc.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-100">{row['Personal F1'].toFixed(3)}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-400">{row['Generic F1'].toFixed(3)}</td>
                        <td className={`px-4 py-3 text-right tabular-nums font-medium ${deltaF1 > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {deltaF1 > 0 ? '+' : ''}{deltaF1.toFixed(3)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <PlotlyChart figure={data.chart} className="rounded-lg border border-zinc-800 overflow-hidden" />
          </Section>
        </>
      )}
    </div>
  )
}
