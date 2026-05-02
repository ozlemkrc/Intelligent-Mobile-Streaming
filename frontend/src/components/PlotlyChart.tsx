import Plot from 'react-plotly.js'
import type { PlotlyFigure } from '../types'

interface Props {
  figure: PlotlyFigure
  className?: string
}

export function PlotlyChart({ figure, className }: Props) {
  return (
    <div className={className}>
      <Plot
        data={figure.data as Plotly.Data[]}
        layout={{
          ...(figure.layout as Partial<Plotly.Layout>),
          autosize: true,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#cbd5e1' },
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        config={{ responsive: true, displayModeBar: 'hover' as const }}
      />
    </div>
  )
}
