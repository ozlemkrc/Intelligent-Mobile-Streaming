import { useState, useEffect, Fragment } from 'react'
import { api } from '../api'
import { useJob } from '../hooks/useJob'
import { Section } from '../components/Section'
import { SliderField } from '../components/SliderField'
import { ProgressBar } from '../components/ProgressBar'
import { MetricCard } from '../components/MetricCard'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure, AppState, AllMetricsResponse } from '../types'

interface Props {
  personas: Record<string, { name: string }>
  appState: AppState
  onStateChange: () => void
}

export function ModelsPage({ personas, appState, onStateChange }: Props) {
  const [lstmWindow, setLstmWindow]     = useState(15)
  const [lstmEpochs, setLstmEpochs]     = useState(25)
  const [lstmPatience, setLstmPatience] = useState(7)
  const [knnK, setKnnK]                 = useState(5)
  const [rfTrees, setRfTrees]           = useState(100)
  const [error, setError]               = useState<string | null>(null)

  const [allMetrics, setAllMetrics]     = useState<AllMetricsResponse | null>(null)
  const [expanded, setExpanded]         = useState<string | null>(null)
  const [expandedCharts, setExpandedCharts] = useState<{ loss: PlotlyFigure; confusion: PlotlyFigure } | null>(null)

  const job = useJob(() => { onStateChange(); loadMetrics() })

  useEffect(() => {
    if (appState.has_persona_models) loadMetrics()
  }, [appState.has_persona_models])

  useEffect(() => {
    if (!expanded) { setExpandedCharts(null); return }
    Promise.all([
      api.getPersonaLossChart(expanded),
      api.getPersonaConfusionChart(expanded),
    ]).then(([loss, confusion]) => {
      setExpandedCharts({ loss: loss as PlotlyFigure, confusion: confusion as PlotlyFigure })
    }).catch(console.error)
  }, [expanded])

  async function loadMetrics() {
    try {
      const d = await api.getAllMetrics()
      setAllMetrics(d)
    } catch { /* not ready */ }
  }

  async function handleTrainAll() {
    setError(null)
    try {
      const { job_id } = await api.trainAllPersonas({
        lstm_window: lstmWindow, lstm_epochs: lstmEpochs, lstm_patience: lstmPatience,
        knn_k: knnK, rf_trees: rfTrees,
      })
      job.start(job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const personaRows = allMetrics
    ? Object.entries(allMetrics.personas).map(([key, m]) => ({ key, ...m }))
    : []

  const avgAccuracy = personaRows.length
    ? personaRows.reduce((s, r) => s + r.accuracy, 0) / personaRows.length
    : null

  const avgF1 = personaRows.length
    ? personaRows.reduce((s, r) => s + r.f1_macro, 0) / personaRows.length
    : null

  return (
    <div className="px-8 py-7 max-w-6xl">

      <Section title="Train Persona Models" subtitle="Trains one LSTM per user on their individual network history">
        <div className="flex flex-wrap items-end gap-5">
          <SliderField label="Window (steps)" value={lstmWindow} min={5} max={30} step={5} onChange={setLstmWindow} className="w-40" />
          <SliderField label="Max epochs" value={lstmEpochs} min={10} max={60} step={5} onChange={setLstmEpochs} className="w-40" />
          <SliderField label="Patience" value={lstmPatience} min={3} max={15} onChange={setLstmPatience} className="w-32" />
          <SliderField label="KNN k" value={knnK} min={1} max={21} step={2} onChange={setKnnK} className="w-28" />
          <SliderField label="RF trees" value={rfTrees} min={50} max={300} step={50} onChange={setRfTrees} className="w-28" />
          <button
            onClick={handleTrainAll}
            disabled={job.isRunning || !appState.has_data}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded transition-colors"
          >
            {job.isRunning ? 'Training…' : 'Train All Personas'}
          </button>
        </div>
        {job.isRunning && (
          <div className="mt-4 w-80">
            <ProgressBar progress={job.progress} label={`Training persona ${Math.ceil(job.progress * Object.keys(personas).length)} / ${Object.keys(personas).length}`} />
          </div>
        )}
        {(error || job.error) && (
          <p className="mt-3 text-sm text-red-400">{error ?? job.error}</p>
        )}
        {!appState.has_data && (
          <p className="mt-3 text-sm text-zinc-500">Generate persona data first.</p>
        )}
      </Section>

      {allMetrics && (
        <>
          <Section title="Trained Models">
            <div className="grid grid-cols-3 gap-3 mb-6">
              <MetricCard label="Mean accuracy" value={avgAccuracy !== null ? avgAccuracy.toFixed(3) : '—'} />
              <MetricCard label="Mean F1 macro" value={avgF1 !== null ? avgF1.toFixed(3) : '—'} />
              <MetricCard label="Models trained" value={`${personaRows.length} / ${Object.keys(personas).length}`} />
            </div>

            <div className="border border-zinc-800 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/60">
                    <th className="text-left px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Persona</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Accuracy</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">F1 Macro</th>
                    <th className="text-right px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">Epochs</th>
                    <th className="w-8" />
                  </tr>
                </thead>
                <tbody>
                  {personaRows.map((row, i) => (
                    <Fragment key={row.key}>
                      <tr
                        onClick={() => setExpanded(expanded === row.key ? null : row.key)}
                        className={`border-b border-zinc-800/60 cursor-pointer transition-colors ${
                          expanded === row.key ? 'bg-zinc-800/60' : i % 2 === 0 ? 'bg-zinc-900/20 hover:bg-zinc-800/30' : 'hover:bg-zinc-800/30'
                        }`}
                      >
                        <td className="px-4 py-3 text-zinc-200 font-medium">{row.persona_name}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-100">{row.accuracy.toFixed(3)}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-100">{row.f1_macro.toFixed(3)}</td>
                        <td className="px-4 py-3 text-right tabular-nums text-zinc-400">{row.epochs_trained}</td>
                        <td className="px-4 py-3 text-zinc-500 text-xs">{expanded === row.key ? '▲' : '▼'}</td>
                      </tr>
                      {expanded === row.key && (
                        <tr className="bg-zinc-800/30 border-b border-zinc-800">
                          <td colSpan={5} className="px-4 py-4">
                            {expandedCharts ? (
                              <div className="grid grid-cols-2 gap-4">
                                <PlotlyChart figure={expandedCharts.loss} className="rounded border border-zinc-700 overflow-hidden" />
                                <PlotlyChart figure={expandedCharts.confusion} className="rounded border border-zinc-700 overflow-hidden" />
                              </div>
                            ) : (
                              <p className="text-sm text-zinc-500">Loading charts…</p>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {Object.keys(allMetrics.baselines).length > 0 && (
            <Section title="Baselines" subtitle="KNN and Random Forest trained on combined data from all personas">
              <div className="grid grid-cols-2 gap-3 w-96">
                {Object.entries(allMetrics.baselines).map(([name, b]) => (
                  <div key={name} className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
                    <div className="text-xs text-zinc-500 uppercase tracking-wide mb-2">{name}</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-zinc-400">Accuracy</span>
                      <span className="tabular-nums text-zinc-100">{b.accuracy.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-sm mt-1">
                      <span className="text-zinc-400">F1 Macro</span>
                      <span className="tabular-nums text-zinc-100">{b.f1_macro.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  )
}
