interface Props {
  progress: number  // 0–1
  label?: string
}

export function ProgressBar({ progress, label }: Props) {
  const pct = Math.round(progress * 100)
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        {label && <span className="text-xs text-zinc-400">{label}</span>}
        <span className="text-xs text-zinc-500 tabular-nums ml-auto">{pct}%</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
