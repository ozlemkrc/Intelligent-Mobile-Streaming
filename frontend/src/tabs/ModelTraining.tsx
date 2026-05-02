import { useState, useEffect } from 'react'
import { api } from '../api'
import { PlotlyChart } from '../components/PlotlyChart'
import { MetricCard } from '../components/MetricCard'
import type { PlotlyFigure } from '../types'

interface Props { hasModels: boolean; activePersona: string; personas: Record<string, { name: string }> }

export function ModelTraining({ hasModels, activePersona, personas }: Props) {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [charts, setCharts] = useState<{
    lstm: PlotlyFigure; baselines: Record<string, PlotlyFigure>; feature_importance: PlotlyFigure
  } | null>(null)
  const [lossChart, setLossChart] = useState<PlotlyFigure | null>(null)

  useEffect(() => {
    if (!hasModels) return
    api.getModelMetrics().then(r => setMetrics(r as Record<string, unknown>)).catch(console.error)
    api.getLossChart().then(r => setLossChart(r as PlotlyFigure)).catch(console.error)
    api.getConfusionCharts().then(r => setCharts(r as typeof charts)).catch(console.error)
  }, [hasModels, activePersona])

  if (!hasModels) return <p className="text-slate-400">Train models from the sidebar first.</p>

  const lstm = metrics?.lstm as Record<string, unknown> | undefined
  const baselines = metrics?.baselines as Record<string, Record<string, unknown>> | undefined

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-100">Model Training</h2>

      {metrics && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">
            Training data — {personas[activePersona]?.name}
          </h3>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <MetricCard label="Total rows" value={(metrics.n_rows as number).toLocaleString()} />
            <MetricCard label="LSTM window" value={String(metrics.window)} />
            <MetricCard label="Epochs trained" value={String(metrics.epochs_trained)} />
          </div>
        </section>
      )}

      {lossChart && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">LSTM Training Curves</h3>
          <PlotlyChart figure={lossChart} className="h-72" />
        </section>
      )}

      {lstm && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">LSTM — Evaluation on Full Persona Dataset</h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <MetricCard label="Accuracy" value={(lstm.accuracy as number).toFixed(3)} />
            <MetricCard label="F1 Macro" value={(lstm.f1_macro as number).toFixed(3)} />
          </div>
          {charts && <PlotlyChart figure={charts.lstm} className="h-80" />}
        </section>
      )}

      {baselines && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">KNN & Random Forest — Baselines</h3>
          <div className="overflow-auto rounded-lg border border-slate-700 mb-4">
            <table className="w-full text-sm">
              <thead className="bg-slate-800">
                <tr>{['Model', 'Accuracy', 'F1 Macro'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                ))}</tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {Object.entries(baselines).map(([name, r]) => (
                  <tr key={name} className="hover:bg-slate-800/50">
                    <td className="px-4 py-2 text-indigo-400">{name}</td>
                    <td className="px-4 py-2 font-mono text-slate-200">{(r.accuracy as number).toFixed(3)}</td>
                    <td className="px-4 py-2 font-mono text-slate-200">{(r.f1_macro as number).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {charts && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(charts.baselines).map(([name, fig]) => (
                <PlotlyChart key={name} figure={fig} className="h-72" />
              ))}
            </div>
          )}
        </section>
      )}

      {charts && (
        <section>
          <h3 className="text-base font-medium text-slate-300 mb-3">Random Forest — Feature Importance</h3>
          <PlotlyChart figure={charts.feature_importance} className="h-72" />
        </section>
      )}
    </div>
  )
}
