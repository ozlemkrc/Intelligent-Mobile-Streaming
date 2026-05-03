interface Props {
  title: string
  subtitle?: string
  action?: React.ReactNode
  children: React.ReactNode
}

export function Section({ title, subtitle, action, children }: Props) {
  return (
    <div className="mb-10">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-500">{title}</h2>
          {subtitle && <p className="text-sm text-zinc-400 mt-0.5">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </div>
  )
}
