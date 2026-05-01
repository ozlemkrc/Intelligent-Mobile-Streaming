from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score

from .streaming_engine import QUALITY_ORDER, compute_metrics


METRIC_LABELS = {
    'qoe_score':        'QoE Score (Mbps-eq.)  ↑ headline',
    'avg_bitrate_mbps': 'Avg Bitrate (Mbps)',
    'rebuffer_events':  'Rebuffering Events',
    'rebuffer_secs':    'Rebuffering Duration (s)',
    'quality_switches': 'Quality Switches',
    'mean_quality_idx': 'Mean Quality Index (0–3)',
    'adapt_accuracy':   'Adaptation Accuracy (%)',
    'quality_mismatch': 'Quality Mismatch (steps)',
}

METHOD_COLORS = {
    'Rule-Based':    '#FFA15A',
    'Threshold':     '#EF553B',
    'ML-Based':      '#636EFA',
}


def compare(
    sim_threshold: pd.DataFrame,
    sim_ml: pd.DataFrame,
    sim_rule: pd.DataFrame | None = None,
) -> dict:
    """Return metric dicts for each method. The rule-based baseline is
    optional so existing two-method calls keep working."""
    m_th = compute_metrics(sim_threshold)
    m_ml = compute_metrics(sim_ml)
    m_th['adapt_accuracy'] *= 100
    m_ml['adapt_accuracy'] *= 100
    out = {}
    if sim_rule is not None:
        m_rule = compute_metrics(sim_rule)
        m_rule['adapt_accuracy'] *= 100
        out['Rule-Based'] = m_rule
    out['Threshold'] = m_th
    out['ML-Based'] = m_ml
    return out


def plot_quality_timeline(
    sim_th: pd.DataFrame,
    sim_ml: pd.DataFrame,
    sim_rule: pd.DataFrame | None = None,
) -> go.Figure:
    q_map = {q: i for i, q in enumerate(QUALITY_ORDER)}
    series = []
    if sim_rule is not None:
        series.append((sim_rule, 'Rule-Based'))
    series.extend([(sim_th, 'Threshold'), (sim_ml, 'ML-Based')])
    titles = {
        'Rule-Based': 'Rule-Based ABR (naive threshold rule)',
        'Threshold':  'Rate-Based ABR (harmonic-mean estimate)',
        'ML-Based':   'ML-Based ABR (estimate × class safety factor)',
    }
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True,
                        subplot_titles=[titles[lbl] for _, lbl in series],
                        vertical_spacing=0.08)

    for row, (sim, label) in enumerate(series, start=1):
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

    for r in range(1, len(series) + 1):
        fig.update_yaxes(
            tickvals=list(range(4)), ticktext=QUALITY_ORDER,
            title_text='Quality', row=r, col=1,
        )
    fig.update_xaxes(title_text='Time (s)', row=len(series), col=1)
    fig.update_layout(height=240 * len(series), showlegend=False,
                      title='Quality Level Over Time  (red shading = rebuffering)')
    return fig


def plot_estimator(
    sim_th: pd.DataFrame,
    sim_ml: pd.DataFrame,
    sim_rule: pd.DataFrame | None = None,
) -> go.Figure:
    """Show actual throughput vs each method's effective (post-safety-factor) signal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_th['time'], y=sim_th['throughput_mbps'],
        mode='lines', name='Actual throughput',
        line=dict(color='#888', width=1, dash='dot'),
    ))
    if sim_rule is not None:
        fig.add_trace(go.Scatter(
            x=sim_rule['time'], y=sim_rule['effective_mbps'],
            mode='lines', name='Rule-Based — last observed throughput',
            line=dict(color=METHOD_COLORS['Rule-Based'], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=sim_th['time'], y=sim_th['effective_mbps'],
        mode='lines', name='Threshold (rate-based) — harmonic estimate',
        line=dict(color=METHOD_COLORS['Threshold'], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sim_ml['time'], y=sim_ml['effective_mbps'],
        mode='lines', name='ML-Based — estimate × safety',
        line=dict(color=METHOD_COLORS['ML-Based'], width=2),
    ))
    fig.update_layout(
        title='Throughput Signal Used for Quality Decisions',
        xaxis_title='Time (s)',
        yaxis_title='Throughput (Mbps)',
        height=300,
    )
    return fig


def plot_buffer(
    sim_th: pd.DataFrame,
    sim_ml: pd.DataFrame,
    sim_rule: pd.DataFrame | None = None,
) -> go.Figure:
    fig = go.Figure()
    series = []
    if sim_rule is not None:
        series.append((sim_rule, 'Rule-Based'))
    series.extend([(sim_th, 'Threshold'), (sim_ml, 'ML-Based')])
    for sim, label in series:
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
    # QoE is the headline metric (col 1); secondary metrics follow.
    bar_metrics = [
        ('qoe_score',        'QoE Score (Mbps-eq.)',    True),
        ('avg_bitrate_mbps', 'Avg Bitrate (Mbps)',      True),
        ('rebuffer_secs',    'Rebuffering Duration (s)', False),
        ('quality_switches', 'Quality Switches',         False),
        ('mean_quality_idx', 'Mean Quality Index',       True),
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

    # Bold the QoE subplot title to signal it is the headline metric.
    fig.layout.annotations[0].update(font=dict(size=13, color='#cdd6f4'))
    fig.update_layout(height=370, title='Method Comparison — QoE (headline) + supporting metrics',
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


# ─────────────────────────────────────────────────────────────────────────────
# Trace-level classifier diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_INT  = {'low': 0, 'medium': 1, 'high': 2}
_CONGESTION_CLASSES = ['low', 'medium', 'high']


def plot_trace_clf_timeline(ts: pd.DataFrame, preds: np.ndarray) -> go.Figure:
    """True vs predicted congestion class over the streaming trace.

    Misclassified time steps are shaded in light red so it is easy to see
    which parts of the trace are difficult for the classifier and whether
    those regions correlate with quality drops or rebuffering.
    """
    true_y = ts['congestion_true'].map(_CLASS_INT)
    pred_y = pd.Series(preds, index=ts.index).map(_CLASS_INT)
    mismatch = ts['congestion_true'].values != preds

    fig = go.Figure()

    # Shade misclassified runs
    in_block = False
    start_t: int = 0
    times = ts['time'].tolist()
    for t, mis in zip(times, mismatch):
        if mis and not in_block:
            start_t = t
            in_block = True
        elif not mis and in_block:
            fig.add_vrect(x0=start_t, x1=t,
                          fillcolor='red', opacity=0.18, line_width=0)
            in_block = False
    if in_block:
        fig.add_vrect(x0=start_t, x1=times[-1],
                      fillcolor='red', opacity=0.18, line_width=0)

    fig.add_trace(go.Scatter(
        x=ts['time'], y=true_y,
        mode='lines', name='True label',
        line=dict(color='#888', width=2, dash='dot'),
        hovertemplate='t=%{x}s  True=%{customdata}',
        customdata=ts['congestion_true'],
    ))
    fig.add_trace(go.Scatter(
        x=ts['time'], y=pred_y,
        mode='lines', name='Predicted',
        line=dict(color=METHOD_COLORS['ML-Based'], width=2),
        hovertemplate='t=%{x}s  Pred=%{customdata}',
        customdata=preds,
    ))
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=['low', 'medium', 'high'],
                     title_text='Congestion Class')
    fig.update_xaxes(title_text='Time (s)')
    fig.update_layout(
        title='Congestion Classification on Streaming Trace  (red = misclassified)',
        height=260,
        legend=dict(orientation='h', y=1.12),
    )
    return fig


def plot_trace_acc_multi_seed(df_runs: pd.DataFrame) -> go.Figure:
    """Box plots of per-seed trace-level accuracy and F1 macro (ML-Based only).

    Complements the i.i.d. test-set scores shown in the ML Training tab:
    this is how well the model classifies the *sequential streaming trace*.
    """
    sub = df_runs[
        (df_runs['method'] == 'ML-Based') & df_runs['trace_accuracy'].notna()
    ]
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=sub['trace_accuracy'], name='Trace Accuracy',
        marker_color=METHOD_COLORS['ML-Based'], boxmean='sd',
    ))
    fig.add_trace(go.Box(
        y=sub['trace_f1'], name='Trace F1 Macro',
        marker_color='#AB63FA', boxmean='sd',
    ))
    fig.update_layout(
        height=320,
        title='Classifier Score on Streaming Trace — Multi-Seed Distribution  (ML-Based)',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed statistical analysis
# ─────────────────────────────────────────────────────────────────────────────
HIGHER_BETTER = {
    'qoe_score':        True,
    'avg_bitrate_mbps': True,
    'rebuffer_events':  False,
    'rebuffer_secs':    False,
    'quality_switches': False,
    'mean_quality_idx': True,
    'adapt_accuracy':   True,
    'quality_mismatch': False,
}


def run_multi_seed(
    trainer,
    persona_key: str,
    n_runs: int = 20,
    duration: int = 180,
    base_seed: int = 1000,
    progress_callback=None,
) -> pd.DataFrame:
    """Run N simulations with different network seeds for a persona; return
    per-run metrics as a long-format DataFrame: seed, method, <metric>...

    Each run uses a different hour-of-day so the trace spans varied conditions
    (home, commute, office) depending on the persona's schedule.
    """
    from .persona_generator import PersonaDataGenerator
    from .streaming_engine import StreamingEngine, compute_metrics

    gen    = PersonaDataGenerator()
    engine = StreamingEngine()
    rows   = []

    for i in range(n_runs):
        seed = base_seed + i
        hour = float(seed % 24)
        ts   = gen.generate_streaming_trace(persona_key, duration=duration,
                                            hour_of_day=hour, seed=seed)
        preds    = trainer.predict_series(ts)
        sim_rule = engine.simulate(ts, method='rule')
        sim_th   = engine.simulate(ts, method='threshold')
        sim_ml   = engine.simulate(ts, method='ml', predictions=preds)
        for method, sim in [('Rule-Based', sim_rule),
                             ('Threshold',  sim_th),
                             ('ML-Based',   sim_ml)]:
            m   = compute_metrics(sim)
            row = {'seed': seed, 'method': method}
            row.update({k: v for k, v in m.items() if k != 'quality_distribution'})
            rows.append(row)
        if progress_callback:
            progress_callback((i + 1) / n_runs)

    return pd.DataFrame(rows)


def plot_multi_seed_bars(df_runs: pd.DataFrame) -> go.Figure:
    """Bar chart with mean ± std error bars across all runs."""
    metrics = [
        ('qoe_score',        'QoE Score (Mbps-eq.)'),
        ('avg_bitrate_mbps', 'Avg Bitrate (Mbps)'),
        ('rebuffer_secs',    'Rebuffer Duration (s)'),
        ('quality_switches', 'Quality Switches'),
        ('mean_quality_idx', 'Mean Quality Index'),
    ]
    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=[m[1] for m in metrics])
    n_runs = int(df_runs['method'].value_counts().iloc[0])
    methods_present = [m for m in ['Rule-Based', 'Threshold', 'ML-Based']
                       if m in df_runs['method'].unique()]
    for col, (key, _) in enumerate(metrics, start=1):
        for method in methods_present:
            sub = df_runs[df_runs['method'] == method][key]
            fig.add_trace(go.Bar(
                name=method, x=[method], y=[sub.mean()],
                error_y=dict(type='data', array=[sub.std()], visible=True),
                marker_color=METHOD_COLORS[method],
                showlegend=(col == 1),
            ), row=1, col=col)
    fig.update_layout(
        height=400, barmode='group',
        title=f'Multi-Seed Comparison — QoE headline + supporting metrics  (mean ± std, n={n_runs} runs per method)',
    )
    return fig


def plot_multi_seed_distribution(df_runs: pd.DataFrame, metric: str) -> go.Figure:
    """Box plot showing the distribution of one metric across runs."""
    fig = go.Figure()
    methods_present = [m for m in ['Rule-Based', 'Threshold', 'ML-Based']
                       if m in df_runs['method'].unique()]
    for method in methods_present:
        sub = df_runs[df_runs['method'] == method]
        fig.add_trace(go.Box(
            y=sub[metric], name=method, marker_color=METHOD_COLORS[method],
            boxmean='sd',
        ))
    fig.update_layout(
        height=320,
        title=f'Distribution of {metric} across runs',
        yaxis_title=metric,
    )
    return fig


def compute_win_rates(df_runs: pd.DataFrame, baseline: str = 'Threshold') -> pd.DataFrame:
    """Per metric: how often ML beats the chosen baseline across seeds.
    `baseline` can be 'Threshold' (rate-based) or 'Rule-Based' (naive)."""
    if baseline not in df_runs['method'].unique():
        raise ValueError(f"Baseline '{baseline}' not present in runs.")
    base = df_runs[df_runs['method'] == baseline].set_index('seed').sort_index()
    ml   = df_runs[df_runs['method'] == 'ML-Based'].set_index('seed').sort_index()
    n = len(base)
    rows = []
    for metric, hb in HIGHER_BETTER.items():
        if hb:
            ml_wins   = int((ml[metric] > base[metric]).sum())
            base_wins = int((ml[metric] < base[metric]).sum())
        else:
            ml_wins   = int((ml[metric] < base[metric]).sum())
            base_wins = int((ml[metric] > base[metric]).sum())
        ties = n - ml_wins - base_wins
        rows.append({
            'Metric':            metric,
            'ML wins':           ml_wins,
            f'{baseline} wins':  base_wins,
            'Ties':              ties,
            'ML win rate':       f"{ml_wins / n * 100:.0f}%",
            'ML mean':           round(ml[metric].mean(), 3),
            f'{baseline} mean':  round(base[metric].mean(), 3),
            f'Δ (ML − {baseline})': round(ml[metric].mean() - base[metric].mean(), 3),
        })
    return pd.DataFrame(rows)
