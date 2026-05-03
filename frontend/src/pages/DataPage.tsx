import { useState, useEffect } from 'react'
import { api } from '../api'
import { useJob } from '../hooks/useJob'
import { Section } from '../components/Section'
import { SliderField } from '../components/SliderField'
import { SelectField } from '../components/SelectField'
import { ProgressBar } from '../components/ProgressBar'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure, AppState } from '../types'

const FEATURES = [
  'throughput_dl', 'throughput_ul', 'latency',
  'packet_loss', 'jitter', 'signal_strength', 'mobility_speed',
]

interface Props {
  personas: Record<string, { name: string }>
  appState: AppState
  onStateChange: () => void
}

interface SummaryRow {
  persona: string; key: string
  low: number; medium: number; high: number; rows: number
}

export function DataPage({ personas, appState, onStateChange }: Props) {
  const [nSessions, setNSessions]     = useState(15)
  const [duration, setDuration]       = useState(300)
  const [seed, setSeed]               = useState(42)
  const [error, setError]             = useState<string | null>(null)

  const [summary, setSummary]         = useState<SummaryRow[] | null>(null)
  const [explorerPersona, setExplorerPersona] = useState('')
  const [explorerFeature, setExplorerFeature] = useState('throughput_dl')
  const [histogram, setHistogram]     = useState<PlotlyFigure | null>(null)
  const [pies, setPies]               = useState<{ location: PlotlyFigure; congestion: PlotlyFigure } | null>(null)

  const job = useJob(() => { onStateChange(); loadSummary() })

  const personaKeys = Object.keys(personas)

  useEffect(() => {
    if (personaKeys.length && !explorerPersona) setExplorerPersona(personaKeys[0])
  }, [personaKeys])

  useEffect(() => {
    if (appState.has_data) loadSummary()
  }, [appState.has_data])

  useEffect(() => {
    if (appState.has_data && explorerPersona) {
      api.getDistributionChart(explorerPersona, explorerFeature).then(d => setHistogram(d as PlotlyFigure))
      api.getPieCharts(explorerPersona).then(d => setPies(d as { location: PlotlyFigure; congestion: PlotlyFigure }))
    }
  }, [explorerPersona, explorerFeature, appState.has_data])

  async function loadSummary() {
    try {
      const d = await api.getDataSummary() as { class_breakdown: SummaryRow[] }
      setSummary(d.class_breakdown)
    } catch { /* ignore if not ready */ }
  }

  async function handleGenerate() {
    setError(null)
    try {
      const { job_id } = await api.generateData({ n_sessions: nSessions, session_duration: duration, seed })
      job.start(job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const personaOptions = personaKeys.map(k => ({ value: k, label: personas[k].name }))
  const featureOptions = FEATURES.map(f => ({ value: f, label: f.replace(/_/g, ' ') }))

  return (
    <div className="px-8 py-7 max-w-6xl">

      <Section title="Generate Training Data">
        <div className="flex flex-wrap items-end gap-5">
          <SliderField label="Sessions per user" value={nSessions} min={5} max={40} onChange={setNSessions} className="w-44" />
          <SliderField label="Session duration (s)" value={duration} min={120} max={600} step={30} onChange={setDuration} className="w-44" />
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-400">Seed</label>
            <input
              type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}
              className="w-20 bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={handleGenerate}
            disabled={job.isRunning}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded transition-colors"
          >
            {job.isRunning ? 'Generating…' : 'Generate All Personas'}
          </button>
        </div>
        {job.isRunning && <div className="mt-4 w-80"><ProgressBar progress={job.progress} label="Generating" /></div>}
        {(error || job.error) && (
          <p className="mt-3 text-sm text-red-400">{error ?? job.error}</p>
        )}
      </Section>

      {summary && (
        <Section title="Persona Overview" subtitle={`${summary.reduce((s, r) => s + r.rows, 0).toLocaleString()} total rows across ${summary.length} personas`}>
          <div className="grid grid-cols-5 gap-3">
            {summary.map(row => (
              <div key={row.key} className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-3">
                <div className="text-sm font-medium text-zinc-200 mb-1">{row.persona}</div>
                <div className="text-xs text-zinc-500 mb-2">{row.rows.toLocaleString()} rows</div>
                <div className="flex h-1.5 rounded-full overflow-hidden gap-px">
                  <div className="bg-emerald-500 rounded-l-full" style={{ width: `${row.low}%` }} title={`Low ${row.low}%`} />
                  <div className="bg-amber-500" style={{ width: `${row.medium}%` }} title={`Med ${row.medium}%`} />
                  <div className="bg-red-500 rounded-r-full" style={{ width: `${row.high}%` }} title={`High ${row.high}%`} />
                </div>
                <div className="flex justify-between text-xs text-zinc-600 mt-1">
                  <span>{row.low}%</span>
                  <span>{row.medium}%</span>
                  <span>{row.high}%</span>
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-3 text-xs text-zinc-500">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />Low</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />Medium</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" />High</span>
          </div>
        </Section>
      )}

      {appState.has_data && personaOptions.length > 0 && (
        <Section title="Data Explorer">
          <div className="flex gap-4 mb-4">
            <SelectField label="Persona" value={explorerPersona} options={personaOptions} onChange={setExplorerPersona} className="w-48" />
            <SelectField label="Feature" value={explorerFeature} options={featureOptions} onChange={setExplorerFeature} className="w-52" />
          </div>
          {histogram && <PlotlyChart figure={histogram} className="rounded-lg border border-zinc-800 overflow-hidden" />}
          {pies && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              <PlotlyChart figure={pies.location} className="rounded-lg border border-zinc-800 overflow-hidden" />
              <PlotlyChart figure={pies.congestion} className="rounded-lg border border-zinc-800 overflow-hidden" />
            </div>
          )}
        </Section>
      )}
    </div>
  )
}
