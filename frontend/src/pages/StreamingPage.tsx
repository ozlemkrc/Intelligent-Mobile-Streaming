import { useState, useEffect } from 'react'
import { api } from '../api'
import { Section } from '../components/Section'
import { SliderField } from '../components/SliderField'
import { SelectField } from '../components/SelectField'
import { MetricCard } from '../components/MetricCard'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure, AppState, SimMetrics } from '../types'

interface Props {
  personas: Record<string, { name: string }>
  appState: AppState
  onStateChange: () => void
}

interface SimResult {
  rule: SimMetrics
  threshold: SimMetrics
  ml: SimMetrics
}

interface Charts {
  throughput: PlotlyFigure
  estimator: PlotlyFigure
  quality_timeline: PlotlyFigure
  buffer: PlotlyFigure
}

const METHOD_LABELS: Record<string, string> = {
  rule: 'Rule-Based',
  threshold: 'Rate-Based',
  ml: 'Personal ML',
}

export function StreamingPage({ personas, appState, onStateChange }: Props) {
  const [personaKey, setPersonaKey] = useState('')
  const [hour, setHour]             = useState(8.5)
  const [duration, setDuration]     = useState(300)
  const [seed, setSeed]             = useState(7)
  const [running, setRunning]       = useState(false)
  const [error, setError]           = useState<string | null>(null)

  const [result, setResult]   = useState<SimResult | null>(null)
  const [charts, setCharts]   = useState<Charts | null>(null)

  const personaKeys = Object.keys(personas)

  useEffect(() => {
    if (personaKeys.length && !personaKey) setPersonaKey(personaKeys[0])
  }, [personaKeys])

  async function handleRun() {
    setError(null)
    setRunning(true)
    try {
      const res = await api.runSimulation({ persona_key: personaKey, hour, duration, seed })
      setResult(res)
      const c = await api.getSimulationCharts() as Charts
      setCharts(c)
      onStateChange()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const personaOptions = personaKeys.map(k => ({ value: k, label: personas[k].name }))
  const canRun = appState.has_models && !running

  return (
    <div className="px-8 py-7 max-w-6xl">

      <Section title="Simulation Setup">
        <div className="flex flex-wrap items-end gap-5">
          <SelectField label="Persona" value={personaKey} options={personaOptions} onChange={setPersonaKey} className="w-48" />
          <SliderField label="Hour of day" value={hour} min={0} max={23} step={0.5} onChange={setHour} className="w-40" />
          <SliderField label="Duration (s)" value={duration} min={60} max={600} step={30} onChange={setDuration} className="w-40" />
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-400">Seed</label>
            <input
              type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}
              className="w-20 bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={handleRun}
            disabled={!canRun}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded transition-colors"
          >
            {running ? 'Running…' : 'Run Simulation'}
          </button>
        </div>
        {!appState.has_models && (
          <p className="mt-3 text-sm text-zinc-500">Train models first.</p>
        )}
        {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
      </Section>

      {result && (
        <>
          <Section title="Results">
            <div className="grid grid-cols-3 gap-4">
              {(['rule', 'threshold', 'ml'] as const).map(method => {
                const m = result[method]
                const isML = method === 'ml'
                const deltaQoe = isML ? result.ml.qoe_score - result.threshold.qoe_score : undefined
                return (
                  <div
                    key={method}
                    className={`border rounded-lg p-4 ${isML ? 'border-blue-700 bg-blue-950/20' : 'border-zinc-800 bg-zinc-900/40'}`}
                  >
                    <div className={`text-xs font-semibold uppercase tracking-widest mb-3 ${isML ? 'text-blue-400' : 'text-zinc-400'}`}>
                      {METHOD_LABELS[method]}
                    </div>
                    <div className="space-y-2.5">
                      <Row label="QoE score" value={m.qoe_score.toFixed(3)} highlight={isML} />
                      <Row label="Avg bitrate" value={`${m.avg_bitrate_mbps.toFixed(2)} Mbps`} />
                      <Row label="Rebuffer events" value={String(m.rebuffer_events)} />
                      <Row label="Rebuffer (s)" value={String(m.rebuffer_secs)} />
                      <Row label="Quality switches" value={String(m.quality_switches)} />
                      <Row label="Adapt accuracy" value={`${(m.adapt_accuracy * 100).toFixed(1)}%`} />
                    </div>
                    {deltaQoe !== undefined && (
                      <div className={`mt-3 pt-3 border-t border-zinc-700/50 text-xs font-mono ${deltaQoe >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {deltaQoe >= 0 ? '+' : ''}{deltaQoe.toFixed(3)} vs rate-based
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Section>

          {charts && (
            <Section title="Timeline Charts">
              <div className="space-y-4">
                <div className="rounded-lg border border-zinc-800 overflow-hidden">
                  <PlotlyChart figure={charts.throughput} />
                </div>
                <div className="rounded-lg border border-zinc-800 overflow-hidden">
                  <PlotlyChart figure={charts.quality_timeline} />
                </div>
                <div className="rounded-lg border border-zinc-800 overflow-hidden">
                  <PlotlyChart figure={charts.buffer} />
                </div>
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  )
}

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-sm tabular-nums font-medium ${highlight ? 'text-zinc-100' : 'text-zinc-300'}`}>{value}</span>
    </div>
  )
}
