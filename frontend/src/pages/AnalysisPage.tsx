import { useState, useEffect } from 'react'
import { api } from '../api'
import { useJob } from '../hooks/useJob'
import { Section } from '../components/Section'
import { SliderField } from '../components/SliderField'
import { SelectField } from '../components/SelectField'
import { ProgressBar } from '../components/ProgressBar'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure, AppState } from '../types'

interface Props {
  personas: Record<string, { name: string }>
  appState: AppState
  onStateChange: () => void
}

interface ComparisonData {
  metrics: Record<string, object>
  bars: PlotlyFigure
  distribution: PlotlyFigure
}

interface MultiSeedData {
  bars: PlotlyFigure
  distribution: PlotlyFigure
  win_rates: Record<string, number | string>[]
}

const METRIC_OPTIONS = [
  { value: 'qoe_score', label: 'QoE Score' },
  { value: 'avg_bitrate_mbps', label: 'Avg Bitrate' },
  { value: 'rebuffer_secs', label: 'Rebuffer (s)' },
  { value: 'quality_switches', label: 'Quality Switches' },
  { value: 'adapt_accuracy', label: 'Adapt Accuracy' },
]

const BASELINE_OPTIONS = [
  { value: 'Rule-Based', label: 'vs Rule-Based' },
  { value: 'Rate-Based', label: 'vs Rate-Based' },
]

export function AnalysisPage({ personas, appState, onStateChange }: Props) {
  const [personaKey, setPersonaKey] = useState('')
  const [nRuns, setNRuns]           = useState(20)
  const [duration, setDuration]     = useState(180)
  const [metric, setMetric]         = useState('qoe_score')
  const [baseline, setBaseline]     = useState('Rule-Based')
  const [error, setError]           = useState<string | null>(null)

  const [comparison, setComparison] = useState<ComparisonData | null>(null)
  const [multiSeed, setMultiSeed]   = useState<MultiSeedData | null>(null)

  const job = useJob(() => { onStateChange(); loadMultiSeed() })

  const personaKeys = Object.keys(personas)

  useEffect(() => {
    if (personaKeys.length && !personaKey) setPersonaKey(personaKeys[0])
  }, [personaKeys])

  useEffect(() => {
    if (appState.has_simulation) {
      api.getComparisonData().then(d => setComparison(d as ComparisonData)).catch(console.error)
    }
  }, [appState.has_simulation])

  useEffect(() => {
    if (appState.has_multiseed) loadMultiSeed()
  }, [appState.has_multiseed, metric, baseline])

  async function loadMultiSeed() {
    try {
      const d = await api.getMultiSeedData(metric, baseline)
      setMultiSeed(d as MultiSeedData)
    } catch { /* not ready */ }
  }

  async function handleRunMultiSeed() {
    setError(null)
    try {
      const { job_id } = await api.runMultiSeed({ persona_key: personaKey, n_runs: nRuns, duration })
      job.start(job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const personaOptions = personaKeys.map(k => ({ value: k, label: personas[k].name }))

  return (
    <div className="px-8 py-7 max-w-6xl">

      {comparison ? (
        <Section title="Single-Seed Comparison" subtitle="From the last simulation run">
          <div className="grid grid-cols-2 gap-4">
            <PlotlyChart figure={comparison.bars} className="rounded-lg border border-zinc-800 overflow-hidden" />
            <PlotlyChart figure={comparison.distribution} className="rounded-lg border border-zinc-800 overflow-hidden" />
          </div>
        </Section>
      ) : (
        <Section title="Single-Seed Comparison">
          <p className="text-sm text-zinc-500">Run a simulation first to see comparison results.</p>
        </Section>
      )}

      <Section title="Multi-Seed Analysis" subtitle="Runs N independent simulations to measure statistical robustness">
        <div className="flex flex-wrap items-end gap-5">
          <SelectField label="Persona" value={personaKey} options={personaOptions} onChange={setPersonaKey} className="w-48" />
          <SliderField label="Runs" value={nRuns} min={5} max={50} step={5} onChange={setNRuns} className="w-32" />
          <SliderField label="Duration (s)" value={duration} min={60} max={300} step={30} onChange={setDuration} className="w-40" />
          <button
            onClick={handleRunMultiSeed}
            disabled={job.isRunning || !appState.has_models}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded transition-colors"
          >
            {job.isRunning ? 'Running…' : 'Run Analysis'}
          </button>
        </div>
        {job.isRunning && <div className="mt-4 w-80"><ProgressBar progress={job.progress} label="Running" /></div>}
        {(error || job.error) && <p className="mt-3 text-sm text-red-400">{error ?? job.error}</p>}
        {!appState.has_models && <p className="mt-3 text-sm text-zinc-500">Train models first.</p>}

        {multiSeed && (
          <div className="mt-6 space-y-4">
            <div className="flex gap-4">
              <SelectField label="Metric" value={metric} options={METRIC_OPTIONS} onChange={setMetric} className="w-48" />
              <SelectField label="Baseline" value={baseline} options={BASELINE_OPTIONS} onChange={setBaseline} className="w-40" />
            </div>

            <PlotlyChart figure={multiSeed.bars} className="rounded-lg border border-zinc-800 overflow-hidden" />
            <PlotlyChart figure={multiSeed.distribution} className="rounded-lg border border-zinc-800 overflow-hidden" />

            {multiSeed.win_rates.length > 0 && (
              <div className="border border-zinc-800 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800 bg-zinc-900/60">
                      {Object.keys(multiSeed.win_rates[0]).map(k => (
                        <th key={k} className="text-left px-4 py-2.5 text-xs font-medium text-zinc-400 uppercase tracking-wide">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {multiSeed.win_rates.map((row, i) => (
                      <tr key={i} className={`border-b border-zinc-800/40 ${i % 2 === 0 ? 'bg-zinc-900/20' : ''}`}>
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-4 py-2.5 text-zinc-200 tabular-nums">
                            {typeof v === 'number' ? v.toFixed(3) : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Section>
    </div>
  )
}
