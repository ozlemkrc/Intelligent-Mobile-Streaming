import { useState, useEffect } from 'react'
import { api } from '../api'
import { PlotlyChart } from '../components/PlotlyChart'
import type { PlotlyFigure } from '../types'

const FEATURES = ['throughput_dl', 'throughput_ul', 'latency', 'packet_loss', 'jitter', 'signal_strength', 'mobility_speed']

interface Props { activePersona: string; hasData: boolean; personas: Record<string, { name: string }> }

export function PersonaData({ activePersona, hasData, personas }: Props) {
  const [summary, setSummary] = useState<{ class_breakdown: Record<string, string | number>[] } | null>(null)
  const [feature, setFeature] = useState('throughput_dl')
  const [distChart, setDistChart] = useState<PlotlyFigure | null>(null)
  const [pieCharts, setPieCharts] = useState<{ location: PlotlyFigure; congestion: PlotlyFigure } | null>(null)
  const [scatterFeatures, setScatterFeatures] = useState(['throughput_dl', 'latency', 'signal_strength', 'packet_loss'])
  const [scatterChart, setScatterChart] = useState<PlotlyFigure | null>(null)

  useEffect(() => {
    if (!hasData) return
    api.getDataSummary().then(r => setSummary(r as typeof summary)).catch(console.error)
    api.getPieCharts(activePersona).then(r => setPieCharts(r as typeof pieCharts)).catch(console.error)
  }, [hasData, activePersona])

  useEffect(() => {
    if (!hasData) return
    api.getDistributionChart(activePersona, feature).then(r => setDistChart(r as PlotlyFigure)).catch(console.error)
  }, [hasData, activePersona, feature])

  useEffect(() => {
    if (!hasData || scatterFeatures.length < 2) return
    api.getScatterChart(activePersona, scatterFeatures.join(','))
      .then(r => setScatterChart(r as PlotlyFigure)).catch(console.error)
  }, [hasData, activePersona, scatterFeatures])

  if (!hasData) return <p className="text-slate-400">Generate data from the sidebar first.</p>

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-100">Persona Data Explorer</h2>

      {summary && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">Congestion Class Breakdown per Persona</h3>
          <div className="overflow-auto rounded-lg border border-slate-700">
            <table className="w-full text-sm">
              <thead className="bg-slate-800">
                <tr>{['Persona', 'Low (%)', 'Medium (%)', 'High (%)', 'Rows'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                ))}</tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {summary.class_breakdown.map((row: Record<string, unknown>, i) => (
                  <tr key={i} className="hover:bg-slate-800/50">
                    <td className="px-4 py-2 text-slate-200">{String(row.persona)}</td>
                    <td className="px-4 py-2 text-emerald-400 font-mono">{String(row.low)}</td>
                    <td className="px-4 py-2 text-amber-400 font-mono">{String(row.medium)}</td>
                    <td className="px-4 py-2 text-red-400 font-mono">{String(row.high)}</td>
                    <td className="px-4 py-2 text-slate-400 font-mono">{String(row.rows)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Distributions differ meaningfully across personas because labels emerge from the physics of each user's locations.
          </p>
        </section>
      )}

      <section>
        <h3 className="text-base font-medium text-slate-300 mb-3">Feature Distributions by Persona</h3>
        <select value={feature} onChange={e => setFeature(e.target.value)}
          className="mb-4 bg-slate-800 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
          {FEATURES.map(f => <option key={f} value={f}>{f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>)}
        </select>
        {distChart && <PlotlyChart figure={distChart} className="h-80" />}
      </section>

      <section>
        <h3 className="text-base font-medium text-slate-300 mb-3">
          Selected Persona — {personas[activePersona]?.name}
        </h3>
        {pieCharts && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PlotlyChart figure={pieCharts.location} className="h-72" />
            <PlotlyChart figure={pieCharts.congestion} className="h-72" />
          </div>
        )}
      </section>

      <section>
        <h3 className="text-base font-medium text-slate-300 mb-3">Pairwise Scatter (500-row sample)</h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {FEATURES.map(f => (
            <label key={f} className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input type="checkbox" checked={scatterFeatures.includes(f)}
                onChange={e => setScatterFeatures(prev =>
                  e.target.checked ? [...prev, f] : prev.filter(x => x !== f)
                )}
                className="accent-indigo-500" />
              <span className="text-slate-300">{f.replace(/_/g, ' ')}</span>
            </label>
          ))}
        </div>
        {scatterChart && scatterFeatures.length >= 2 && <PlotlyChart figure={scatterChart} className="h-[560px]" />}
      </section>
    </div>
  )
}
