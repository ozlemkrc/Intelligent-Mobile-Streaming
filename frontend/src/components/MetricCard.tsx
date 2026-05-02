interface Props {
  label: string
  value: string | number
  delta?: number
  unit?: string
}

export function MetricCard({ label, value, delta, unit }: Props) {
  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
      <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-mono font-semibold text-slate-100">
        {value}{unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
      </div>
      {delta !== undefined && (
        <div className={`text-sm mt-1 font-mono ${delta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
        </div>
      )}
    </div>
  )
}
