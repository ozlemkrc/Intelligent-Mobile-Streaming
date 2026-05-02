interface Props {
  progress: number
  label?: string
}

export function ProgressBar({ progress, label }: Props) {
  const pct = Math.round(progress * 100)
  return (
    <div className="space-y-1">
      {label && <div className="text-sm text-slate-400">{label} {pct}%</div>}
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
