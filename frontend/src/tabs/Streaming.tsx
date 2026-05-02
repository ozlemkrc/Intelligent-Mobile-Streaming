import { useState, useEffect } from 'react'
import { api } from '../api'
import { PlotlyChart } from '../components/PlotlyChart'
import { MetricCard } from '../components/MetricCard'
import type { PlotlyFigure } from '../types'

export interface SimMetrics {
  qoe_score: number; avg_bitrate_mbps: number; rebuffer_events: number
  rebuffer_secs: number; quality_switches: number; adapt_accuracy: number
}

interface Props {
  hasSimulation: boolean
  simMetrics: { rule: SimMetrics; threshold: SimMetrics; ml: SimMetrics } | null
}

export function Streaming({ hasSimulation, simMetrics }: Props) {
  const [charts, setCharts] = useState<{
    throughput: PlotlyFigure; estimator: PlotlyFigure; quality_timeline: PlotlyFigure; buffer: PlotlyFigure
  } | null>(null)

  useEffect(() => {
    if (!hasSimulation) return
    api.getSimulationCharts().then(r => setCharts(r as typeof charts)).catch(console.error)
  }, [hasSimulation])

  if (!hasSimulation) return <p className="text-slate-400">Run a simulation from the sidebar first.</p>

  const methods = simMetrics
    ? ([
        ['Rule-Based', 'naive threshold', simMetrics.rule],
        ['Rate-Based', 'harmonic-mean estimate', simMetrics.threshold],
        ['Personal-ML', 'estimate × LSTM safety', simMetrics.ml],
      ] as const)
    : []

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-100">Adaptive Bitrate Streaming Simulation</h2>

      {simMetrics && (
        <>
          <section>
            <h3 className="text-base font-medium text-slate-300 mb-1">
              QoE Score <span className="text-slate-500 text-sm font-normal">(avg bitrate − 0.3·rebuffer_secs − 0.02·switches, higher is better)</span>
            </h3>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <MetricCard label="Rule-Based QoE" value={simMetrics.rule.qoe_score.toFixed(3)} />
              <MetricCard label="Rate-Based QoE" value={simMetrics.threshold.qoe_score.toFixed(3)}
                delta={simMetrics.threshold.qoe_score - simMetrics.rule.qoe_score} />
              <MetricCard label="Personal-ML QoE" value={simMetrics.ml.qoe_score.toFixed(3)}
                delta={simMetrics.ml.qoe_score - simMetrics.rule.qoe_score} />
            </div>
          </section>

          <section>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {methods.map(([label, sub, m]) => (
                <div key={label} className="bg-slate-800 border border-slate-700 rounded-lg p-4 space-y-3">
                  <div>
                    <div className="font-medium text-slate-200">{label}</div>
                    <div className="text-xs text-slate-500">{sub}</div>
                  </div>
                  {[
                    ['Avg Bitrate', `${m.avg_bitrate_mbps.toFixed(3)} Mbps`],
                    ['Rebuffering Events', String(m.rebuffer_events)],
                    ['Rebuffering (s)', String(m.rebuffer_secs)],
                    ['Quality Switches', String(m.quality_switches)],
                    ['Adaptation Accuracy', `${(m.adapt_accuracy * 100).toFixed(1)}%`],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-slate-400">{k}</span>
                      <span className="font-mono text-slate-200">{v}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {charts && (
        <section className="space-y-4">
          <PlotlyChart figure={charts.throughput} className="h-72" />
          <PlotlyChart figure={charts.estimator} className="h-72" />
          <PlotlyChart figure={charts.quality_timeline} className="h-72" />
          <PlotlyChart figure={charts.buffer} className="h-72" />
        </section>
      )}
    </div>
  )
}
