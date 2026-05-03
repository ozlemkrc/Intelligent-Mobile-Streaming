interface Props {
  label: string
  value: string | number
  sub?: string
  positive?: boolean
  negative?: boolean
}

export function MetricCard({ label, value, sub, positive, negative }: Props) {
  const valueColor = positive
    ? 'text-emerald-400'
    : negative
    ? 'text-red-400'
    : 'text-zinc-100'

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
      <div className={`text-2xl font-semibold tabular-nums ${valueColor}`}>{value}</div>
      {sub && <div className="text-xs text-zinc-500 mt-0.5">{sub}</div>}
      <div className="text-xs text-zinc-500 mt-1.5 uppercase tracking-wide">{label}</div>
    </div>
  )
}
