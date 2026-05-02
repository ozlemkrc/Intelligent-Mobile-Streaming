import { useState, useEffect } from 'react'
import { api } from '../api'
import { PlotlyChart } from '../components/PlotlyChart'
import { ProgressBar } from '../components/ProgressBar'
import { useJob } from '../hooks/useJob'
import type { PlotlyFigure } from '../types'

const METRIC_LABELS: Record<string, string> = {
  qoe_score: 'QoE Score (Mbps-eq.) ↑', avg_bitrate_mbps: 'Avg Bitrate (Mbps) ↑',
  rebuffer_events: 'Rebuffering Events ↓', rebuffer_secs: 'Rebuffering Duration (s) ↓',
  quality_switches: 'Quality Switches ↓', mean_quality_idx: 'Mean Quality Index ↑',
  adapt_accuracy: 'Adaptation Accuracy ↑', quality_mismatch: 'Quality Mismatch ↓',
}
const MS_METRICS = ['qoe_score', 'avg_bitrate_mbps', 'rebuffer_events', 'rebuffer_secs',
  'quality_switches', 'mean_quality_idx', 'adapt_accuracy']

interface Props { hasSimulation: boolean; activePersona: string }

export function Comparison({ hasSimulation, activePersona }: Props) {
  const [cmpData, setCmpData] = useState<{
    metrics: Record<string, Record<string, unknown>>
    bars: PlotlyFigure; distribution: PlotlyFigure
  } | null>(null)
  const [msNRuns, setMsNRuns] = useState(20)
  const [msDuration, setMsDuration] = useState(180)
  const [msBaseline, setMsBaseline] = useState('Rule-Based')
  const [msMetric, setMsMetric] = useState('qoe_score')
  const [msData, setMsData] = useState<{
    bars: PlotlyFigure; distribution: PlotlyFigure; win_rates: Record<string, unknown>[]
  } | null>(null)

  const job = useJob(() => {
    api.getMultiSeedData(msMetric, msBaseline).then(r => setMsData(r as typeof msData)).catch(console.error)
  })

  useEffect(() => {
    if (!hasSimulation) return
    api.getComparisonData().then(r => setCmpData(r as typeof cmpData)).catch(console.error)
  }, [hasSimulation])

  useEffect(() => {
    if (!msData) return
    api.getMultiSeedData(msMetric, msBaseline).then(r => setMsData(r as typeof msData)).catch(console.error)
  }, [msMetric, msBaseline])

  if (!hasSimulation) return <p className="text-slate-400">Run a simulation from the sidebar first.</p>

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-100">Performance Comparison</h2>

      {cmpData && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PlotlyChart figure={cmpData.bars} className="h-80" />
            <PlotlyChart figure={cmpData.distribution} className="h-80" />
          </div>

          <section>
            <h3 className="text-base font-medium text-slate-300 mb-3">Metric Table</h3>
            <div className="overflow-auto rounded-lg border border-slate-700">
              <table className="w-full text-sm">
                <thead className="bg-slate-800">
                  <tr>{['Metric', 'Rule-Based', 'Rate-Based', 'ML-Based'].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                  ))}</tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {Object.entries(METRIC_LABELS).map(([key, label]) => {
                    const m = cmpData.metrics
                    const vals = [m['Rule-Based']?.[key], m['Threshold']?.[key], m['ML-Based']?.[key]]
                    return (
                      <tr key={key} className="hover:bg-slate-800/50">
                        <td className="px-4 py-2 text-slate-300">{label}</td>
                        {vals.map((v, i) => (
                          <td key={i} className="px-4 py-2 font-mono text-slate-200">
                            {typeof v === 'number' ? v.toFixed(3) : String(v ?? '—')}
                          </td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* Multi-seed */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-200">Statistical Robustness — Multi-Seed Analysis</h3>
        <p className="text-sm text-slate-400">
          Run N independent simulations at different hours and compare mean ± std and win-rates per metric across methods.
        </p>
        <div className="grid grid-cols-2 gap-4 max-w-sm">
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Number of runs</span>
              <span className="text-indigo-400 font-mono">{msNRuns}</span>
            </div>
            <input type="range" min={5} max={50} step={5} value={msNRuns}
              onChange={e => setMsNRuns(Number(e.target.value))} className="w-full h-1.5 rounded-full bg-slate-700 cursor-pointer" />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Duration per run (s)</span>
              <span className="text-indigo-400 font-mono">{msDuration}</span>
            </div>
            <input type="range" min={60} max={300} step={30} value={msDuration}
              onChange={e => setMsDuration(Number(e.target.value))} className="w-full h-1.5 rounded-full bg-slate-700 cursor-pointer" />
          </div>
        </div>
        <button
          onClick={async () => {
            const { job_id } = await api.runMultiSeed({ persona_key: activePersona, n_runs: msNRuns, duration: msDuration })
            job.start(job_id)
          }}
          disabled={job.isRunning}
          className="py-2 px-4 rounded-md text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-40"
        >
          Run Multi-Seed Analysis
        </button>
        {job.isRunning && <ProgressBar progress={job.progress} label="Running…" />}
        {job.error && <p className="text-red-400 text-sm">{job.error}</p>}

        {msData && (
          <div className="space-y-4">
            <PlotlyChart figure={msData.bars} className="h-80" />
            <div className="flex gap-4 flex-wrap">
              <div className="space-y-1">
                <div className="text-xs text-slate-400">Win-rate baseline</div>
                <select value={msBaseline} onChange={e => setMsBaseline(e.target.value)}
                  className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
                  {['Rule-Based', 'Threshold'].map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-slate-400">Distribution metric</div>
                <select value={msMetric} onChange={e => setMsMetric(e.target.value)}
                  className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
                  {MS_METRICS.map(m => <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>)}
                </select>
              </div>
            </div>

            {msData.win_rates.length > 0 && (
              <div className="overflow-auto rounded-lg border border-slate-700">
                <table className="w-full text-sm">
                  <thead className="bg-slate-800">
                    <tr>{Object.keys(msData.win_rates[0]).map(h => (
                      <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50">
                    {msData.win_rates.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/50">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-4 py-2 text-slate-200 font-mono">
                            {typeof v === 'number' ? v.toFixed(3) : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <PlotlyChart figure={msData.distribution} className="h-80" />
          </div>
        )}
      </section>
    </div>
  )
}
