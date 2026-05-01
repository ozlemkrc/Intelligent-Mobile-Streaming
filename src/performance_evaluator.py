from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .streaming_engine import QUALITY_ORDER, compute_metrics


METRIC_LABELS = {
    'rebuffer_events':  'Rebuffering Events',
    'rebuffer_secs':    'Rebuffering Duration (s)',
    'quality_switches': 'Quality Switches',
    'mean_quality_idx': 'Mean Quality Index (0–3)',
    'adapt_accuracy':   'Adaptation Accuracy (%)',
    'quality_mismatch': 'Quality Mismatch (steps)',
}

METHOD_COLORS = {'Threshold': '#EF553B', 'ML-Based': '#636EFA'}


def compare(sim_threshold: pd.DataFrame, sim_ml: pd.DataFrame) -> dict:
    """Return metric dicts for both methods."""
    m_th = compute_metrics(sim_threshold)
    m_ml = compute_metrics(sim_ml)
    m_ml['adapt_accuracy'] *= 100
    m_th['adapt_accuracy'] *= 100
    return {'Threshold': m_th, 'ML-Based': m_ml}


def plot_quality_timeline(sim_th: pd.DataFrame, sim_ml: pd.DataFrame) -> go.Figure:
    q_map = {q: i for i, q in enumerate(QUALITY_ORDER)}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Threshold-Based ABR', 'ML-Based ABR'],
                        vertical_spacing=0.08)

    for row, (sim, label) in enumerate([(sim_th, 'Threshold'), (sim_ml, 'ML-Based')], start=1):
        y = sim['quality'].map(q_map)
        fig.add_trace(go.Scatter(
            x=sim['time'], y=y,
            mode='lines', name=label,
            line=dict(color=METHOD_COLORS[label], width=2),
            hovertemplate='t=%{x}s<br>Quality=%{customdata}',
            customdata=sim['quality'],
        ), row=row, col=1)
        # Shade rebuffering regions
        reb = sim[sim['rebuffering']]
        if not reb.empty:
            for _, grp in reb.groupby((reb['time'].diff() != 1).cumsum()):
                fig.add_vrect(
                    x0=grp['time'].iloc[0], x1=grp['time'].iloc[-1],
                    fillcolor='red', opacity=0.15, line_width=0,
                    row=row, col=1,
                )

    fig.update_yaxes(
        tickvals=list(range(4)), ticktext=QUALITY_ORDER,
        title_text='Quality', row=1, col=1,
    )
    fig.update_yaxes(
        tickvals=list(range(4)), ticktext=QUALITY_ORDER,
        title_text='Quality', row=2, col=1,
    )
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_layout(height=480, showlegend=False,
                      title='Quality Level Over Time  (red shading = rebuffering)')
    return fig


def plot_buffer(sim_th: pd.DataFrame, sim_ml: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for sim, label in [(sim_th, 'Threshold'), (sim_ml, 'ML-Based')]:
        fig.add_trace(go.Scatter(
            x=sim['time'], y=sim['buffer_s'],
            mode='lines', name=label,
            line=dict(color=METHOD_COLORS[label], width=2),
        ))
    fig.update_layout(
        title='Buffer Level Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Buffer (s)',
        height=300,
    )
    return fig


def plot_throughput(ts: pd.DataFrame) -> go.Figure:
    color_map = {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts['time'], y=ts['throughput_dl'],
        mode='lines', name='Throughput DL',
        line=dict(color='steelblue', width=1.5),
    ))
    # Shade by true congestion
    prev_label = ts['congestion_true'].iloc[0]
    start_t = 0
    for i, row in ts.iterrows():
        if row['congestion_true'] != prev_label or i == len(ts) - 1:
            fig.add_vrect(
                x0=start_t, x1=row['time'],
                fillcolor=color_map.get(prev_label, 'gray'),
                opacity=0.12, line_width=0,
                annotation_text=prev_label, annotation_position='top left',
                annotation_font_size=10,
            )
            start_t = row['time']
            prev_label = row['congestion_true']
    fig.update_layout(
        title='Network Throughput (shaded by congestion zone)',
        xaxis_title='Time (s)',
        yaxis_title='DL Throughput (Mbps)',
        height=280,
    )
    return fig


def plot_comparison_bars(metrics: dict) -> go.Figure:
    bar_metrics = [
        ('rebuffer_events',  'Rebuffering Events', False),
        ('rebuffer_secs',    'Rebuffering Duration (s)', False),
        ('quality_switches', 'Quality Switches', False),
        ('mean_quality_idx', 'Mean Quality Index', True),
        ('adapt_accuracy',   'Adaptation Accuracy (%)', True),
    ]
    fig = make_subplots(
        rows=1, cols=len(bar_metrics),
        subplot_titles=[b[1] for b in bar_metrics],
    )
    for col, (key, label, higher_better) in enumerate(bar_metrics, start=1):
        for method, m in metrics.items():
            fig.add_trace(go.Bar(
                name=method, x=[method], y=[m[key]],
                marker_color=METHOD_COLORS[method],
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.update_layout(height=350, title='Method Comparison — Key Metrics',
                      barmode='group')
    return fig


def plot_quality_distribution(metrics: dict) -> go.Figure:
    fig = go.Figure()
    for method, m in metrics.items():
        dist = m['quality_distribution']
        fig.add_trace(go.Bar(
            name=method,
            x=QUALITY_ORDER,
            y=[dist.get(q, 0) * 100 for q in QUALITY_ORDER],
            marker_color=METHOD_COLORS[method],
        ))
    fig.update_layout(
        barmode='group',
        title='Time Spent at Each Quality Level (%)',
        xaxis_title='Quality Level',
        yaxis_title='% of Session',
        height=320,
    )
    return fig
