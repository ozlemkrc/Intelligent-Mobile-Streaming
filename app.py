"""
Intelligent Mobile Streaming — CSE 476 Term Project
Streamlit dashboard: Network Simulation · ML Training · Streaming · Comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.network_simulator import NetworkSimulator
from src.ml_classifier import FEATURES, CLASSES, NetworkClassifier, lstm_available
from src.streaming_engine import StreamingEngine, QUALITY_ORDER, compute_metrics
from src.performance_evaluator import (
    compare, plot_quality_timeline, plot_buffer,
    plot_throughput, plot_comparison_bars, plot_quality_distribution,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Mobile Streaming",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e1e2e; border-radius: 8px; padding: 1rem;
        text-align: center; margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #cdd6f4; }
    .metric-label { font-size: 0.8rem; color: #a6adc8; }
    .better  { color: #a6e3a1; }
    .worse   { color: #f38ba8; }
    .neutral { color: #cba6f7; }
    h1 { color: #cdd6f4; }
    /* Fix tab height so labels aren't clipped */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 16px;
        line-height: 48px;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"] > div {
        line-height: normal;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for key in ['dataset', 'classifier', 'ts', 'sim_th', 'sim_ml', 'ml_preds']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 IMS Dashboard")

    st.divider()

    st.subheader("1 · Dataset")
    n_per_class = st.slider("Samples per class", 100, 1000, 400, 100)
    rng_seed = st.number_input("Random seed", 0, 9999, 42)

    if st.button("Generate Dataset", use_container_width=True, type="primary"):
        sim = NetworkSimulator()
        st.session_state.dataset = sim.generate_dataset(n_per_class, rng_seed)
        st.session_state.classifier = None   # invalidate trained model
        st.success(f"Generated {len(st.session_state.dataset)} samples.")

    st.divider()
    st.subheader("2 · Model Training")
    knn_k      = st.slider("KNN — k neighbours", 1, 21, 5, 2)
    rf_trees   = st.slider("RF — n estimators", 50, 500, 100, 50)
    test_split = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)

    train_disabled = st.session_state.dataset is None
    if st.button("Train Models", use_container_width=True, type="primary",
                 disabled=train_disabled):
        clf = NetworkClassifier(knn_k=knn_k, rf_trees=rf_trees, random_state=int(rng_seed))
        with st.spinner("Training KNN & Random Forest …"):
            clf.train(st.session_state.dataset, test_size=test_split)
        st.session_state.classifier = clf
        st.success("Models trained.")

    st.divider()
    st.subheader("3 · Streaming Simulation")
    sim_duration = st.slider("Duration (seconds)", 60, 600, 300, 30)
    sim_seed     = st.number_input("Simulation seed", 0, 9999, 7)
    ml_model_name = st.selectbox("Active ML model", ['Random Forest', 'KNN'])

    sim_disabled = st.session_state.classifier is None
    if st.button("Run Simulation", use_container_width=True, type="primary",
                 disabled=sim_disabled):
        ns = NetworkSimulator()
        ts = ns.generate_time_series(sim_duration, int(sim_seed))
        st.session_state.ts = ts

        clf = st.session_state.classifier
        ml_preds = clf.predict_series(ts, model_name=ml_model_name)
        st.session_state.ml_preds = ml_preds

        engine = StreamingEngine()
        st.session_state.sim_th = engine.simulate(ts, method='threshold')
        st.session_state.sim_ml = engine.simulate(ts, method='ml', predictions=ml_preds)
        st.success("Simulation complete.")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Network Data",
    "🤖 ML Training",
    "▶️ Streaming",
    "📈 Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.title("Intelligent Mobile Streaming")
    st.markdown("""
    ### What this system does

    | Step | Module | Description |
    |------|--------|-------------|
    | 1 | **Network Simulator** | Generates synthetic mobile network traces (throughput, latency, packet loss, jitter, signal strength, mobility speed) for three congestion levels |
    | 2 | **ML Classifier** | Trains K-Nearest Neighbours and Random Forest models to predict network congestion class from raw measurements |
    | 3 | **Streaming Engine** | Simulates an ABR video session using (a) simple threshold rules and (b) ML-predicted congestion class |
    | 4 | **Performance Evaluator** | Compares both methods across rebuffering events, average quality, quality stability, and adaptation accuracy |

    ### Network parameters modelled
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - **Throughput DL / UL** (Mbps)
        - **Latency** (ms)
        - **Packet Loss** (%)
        """)
    with col_b:
        st.markdown("""
        - **Jitter** (ms)
        - **Signal Strength** (dBm)
        - **Mobility Speed** (km/h)
        """)

    st.markdown("""
    ### Quality levels

    | Label | Bitrate |
    |-------|---------|
    | 240p  | 300 kbps |
    | 480p  | 1 000 kbps |
    | 720p  | 2 500 kbps |
    | 1080p | 5 000 kbps |

    ### Quick start
    1. **Generate Dataset** in the sidebar → explore distributions in *Network Data*
    2. **Train Models** → inspect accuracy and confusion matrices in *ML Training*
    3. **Run Simulation** → watch quality timelines in *Streaming*
    4. Switch to **Comparison** for the full performance breakdown
    """)

    if lstm_available():
        st.info("TensorFlow detected — LSTM model is available in the ML Training tab.")
    else:
        st.warning("TensorFlow not installed — LSTM is disabled. `pip install tensorflow` to enable it.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Network Data
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Network Data Explorer")

    if st.session_state.dataset is None:
        st.info("Generate a dataset from the sidebar first.")
    else:
        df = st.session_state.dataset
        n_low    = (df.congestion == 'low').sum()
        n_med    = (df.congestion == 'medium').sum()
        n_high   = (df.congestion == 'high').sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", len(df))
        c2.metric("Low Congestion", n_low)
        c3.metric("Medium Congestion", n_med)
        c4.metric("High Congestion", n_high)

        st.subheader("Feature Distributions by Congestion Class")
        feat_select = st.selectbox("Select feature", FEATURES)
        fig_hist = px.histogram(
            df, x=feat_select, color='congestion', barmode='overlay',
            color_discrete_map={'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'},
            nbins=50, opacity=0.75,
            labels={feat_select: feat_select.replace('_', ' ').title()},
        )
        fig_hist.update_layout(height=340)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        corr = df[FEATURES].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, aspect='auto',
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Pairwise Scatter (sample)")
        feat_pair = st.multiselect(
            "Features for scatter matrix",
            FEATURES,
            default=['throughput_dl', 'latency', 'packet_loss', 'signal_strength'],
        )
        if len(feat_pair) >= 2:
            sample = df.sample(min(500, len(df)), random_state=1)
            fig_pair = px.scatter_matrix(
                sample, dimensions=feat_pair, color='congestion',
                color_discrete_map={'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'},
            )
            fig_pair.update_traces(diagonal_visible=False, marker_size=3)
            fig_pair.update_layout(height=560)
            st.plotly_chart(fig_pair, use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df.head(100), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Training
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Machine Learning — Training & Evaluation")

    if st.session_state.classifier is None:
        st.info("Train models from the sidebar first.")
    else:
        clf: NetworkClassifier = st.session_state.classifier
        results = clf.results

        # ── Summary table ──
        st.subheader("Model Performance Summary")
        summary_rows = []
        for name, r in results.items():
            rep = r['report']
            summary_rows.append({
                'Model':     name,
                'Accuracy':  f"{r['accuracy']:.3f}",
                'F1 Macro':  f"{r['f1_macro']:.3f}",
                'Prec (low)':    f"{rep['low']['precision']:.3f}",
                'Prec (med)':    f"{rep['medium']['precision']:.3f}",
                'Prec (high)':   f"{rep['high']['precision']:.3f}",
                'Recall (low)':  f"{rep['low']['recall']:.3f}",
                'Recall (med)':  f"{rep['medium']['recall']:.3f}",
                'Recall (high)': f"{rep['high']['recall']:.3f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # ── Confusion matrices ──
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(len(results))
        for col, (name, r) in zip(cm_cols, results.items()):
            cm = r['confusion_matrix']
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=CLASSES, y=CLASSES,
                color_continuous_scale='Blues',
                labels={'x': 'Predicted', 'y': 'Actual'},
                title=name,
            )
            fig_cm.update_layout(height=320, coloraxis_showscale=False)
            col.plotly_chart(fig_cm, use_container_width=True)

        # ── Feature importance ──
        st.subheader("Random Forest — Feature Importance")
        fi = clf.feature_importances_
        fig_fi = px.bar(
            x=list(fi.values()), y=list(fi.keys()),
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=list(fi.values()),
            color_continuous_scale='Teal',
        )
        fig_fi.update_layout(height=320, coloraxis_showscale=False,
                             yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── LSTM (optional) ──
        if lstm_available():
            st.subheader("LSTM — Time-Series Classifier")
            if st.session_state.dataset is not None:
                lstm_epochs = st.slider("LSTM epochs", 5, 50, 20, 5)
                lstm_window = st.slider("LSTM window (time steps)", 5, 30, 10, 5)
                if st.button("Train LSTM", type="primary"):
                    from src.ml_classifier import LSTMClassifier
                    lstm_clf = LSTMClassifier(window=lstm_window, epochs=lstm_epochs)
                    with st.spinner("Training LSTM …"):
                        lstm_res = lstm_clf.train(st.session_state.dataset)
                    st.success(
                        f"LSTM — Accuracy: {lstm_res['accuracy']:.3f}  |  "
                        f"F1 Macro: {lstm_res['f1_macro']:.3f}"
                    )
                    cm = lstm_res['confusion_matrix']
                    fig_lstm = px.imshow(
                        cm, text_auto=True, x=CLASSES, y=CLASSES,
                        color_continuous_scale='Blues',
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        title='LSTM Confusion Matrix',
                    )
                    fig_lstm.update_layout(height=320, coloraxis_showscale=False)
                    st.plotly_chart(fig_lstm, use_container_width=True)
        else:
            st.info("Install TensorFlow to enable the LSTM classifier.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Streaming Simulation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Adaptive Bitrate Streaming Simulation")

    if st.session_state.sim_th is None:
        st.info("Run a simulation from the sidebar first.")
    else:
        ts     = st.session_state.ts
        sim_th = st.session_state.sim_th
        sim_ml = st.session_state.sim_ml

        # Quick metrics
        m_th = compute_metrics(sim_th)
        m_ml = compute_metrics(sim_ml)

        def _delta(val_ml, val_th, lower_better=True):
            delta = val_ml - val_th
            if lower_better:
                css = "better" if delta < 0 else ("worse" if delta > 0 else "neutral")
            else:
                css = "better" if delta > 0 else ("worse" if delta < 0 else "neutral")
            sign = "+" if delta > 0 else ""
            return f'<span class="{css}">{sign}{delta:.2f}</span>'

        col_th, col_ml = st.columns(2)
        with col_th:
            st.markdown("#### Threshold-Based")
            st.metric("Rebuffering Events",   m_th['rebuffer_events'])
            st.metric("Rebuffering Duration", f"{m_th['rebuffer_secs']}s")
            st.metric("Quality Switches",     m_th['quality_switches'])
            st.metric("Mean Quality Index",   f"{m_th['mean_quality_idx']:.2f} / 3")
            st.metric("Adaptation Accuracy",  f"{m_th['adapt_accuracy']*100:.1f}%")

        with col_ml:
            st.markdown("#### ML-Based")
            st.metric("Rebuffering Events",
                      m_ml['rebuffer_events'],
                      delta=m_ml['rebuffer_events'] - m_th['rebuffer_events'],
                      delta_color="inverse")
            st.metric("Rebuffering Duration",
                      f"{m_ml['rebuffer_secs']}s",
                      delta=m_ml['rebuffer_secs'] - m_th['rebuffer_secs'],
                      delta_color="inverse")
            st.metric("Quality Switches",
                      m_ml['quality_switches'],
                      delta=m_ml['quality_switches'] - m_th['quality_switches'],
                      delta_color="inverse")
            st.metric("Mean Quality Index",
                      f"{m_ml['mean_quality_idx']:.2f} / 3",
                      delta=round(m_ml['mean_quality_idx'] - m_th['mean_quality_idx'], 3))
            st.metric("Adaptation Accuracy",
                      f"{m_ml['adapt_accuracy']*100:.1f}%",
                      delta=f"{(m_ml['adapt_accuracy'] - m_th['adapt_accuracy'])*100:.1f}%")

        st.divider()

        # Network throughput timeline
        st.plotly_chart(plot_throughput(ts), use_container_width=True)

        # Quality timelines
        st.plotly_chart(plot_quality_timeline(sim_th, sim_ml), use_container_width=True)

        # Buffer levels
        st.plotly_chart(plot_buffer(sim_th, sim_ml), use_container_width=True)

        with st.expander("Raw simulation data"):
            view = st.radio("Show data for:", ['Threshold', 'ML-Based'], horizontal=True)
            df_show = sim_th if view == 'Threshold' else sim_ml
            st.dataframe(df_show, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Performance Comparison")

    if st.session_state.sim_th is None:
        st.info("Run a simulation from the sidebar first.")
    else:
        sim_th = st.session_state.sim_th
        sim_ml = st.session_state.sim_ml
        metrics = compare(sim_th, sim_ml)

        st.plotly_chart(plot_comparison_bars(metrics), use_container_width=True)
        st.plotly_chart(plot_quality_distribution(metrics), use_container_width=True)

        st.subheader("Metric Table")
        metric_rows = []
        for key, label in [
            ('rebuffer_events',  'Rebuffering Events   (↓ better)'),
            ('rebuffer_secs',    'Rebuffering Dur. (s) (↓ better)'),
            ('quality_switches', 'Quality Switches     (↓ better)'),
            ('mean_quality_idx', 'Mean Quality Index   (↑ better)'),
            ('adapt_accuracy',   'Adaptation Accuracy %(↑ better)'),
            ('quality_mismatch', 'Quality Mismatch     (↓ better)'),
        ]:
            th_val = metrics['Threshold'][key]
            ml_val = metrics['ML-Based'][key]
            metric_rows.append({'Metric': label,
                                 'Threshold': th_val,
                                 'ML-Based':  ml_val})
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # Narrative conclusion
        st.subheader("Analysis")
        m_th = metrics['Threshold']
        m_ml = metrics['ML-Based']

        rb_diff    = m_th['rebuffer_events'] - m_ml['rebuffer_events']
        sw_diff    = m_th['quality_switches'] - m_ml['quality_switches']
        acc_diff   = m_ml['adapt_accuracy'] - m_th['adapt_accuracy']
        qual_diff  = m_ml['mean_quality_idx'] - m_th['mean_quality_idx']

        lines = []
        if rb_diff > 0:
            lines.append(f"- ML-Based caused **{rb_diff} fewer rebuffering events**, improving playback continuity.")
        elif rb_diff < 0:
            lines.append(f"- Threshold caused **{-rb_diff} fewer rebuffering events** in this run.")
        else:
            lines.append("- Both methods experienced the same number of rebuffering events.")

        if sw_diff > 0:
            lines.append(f"- ML-Based made **{sw_diff} fewer quality switches**, indicating higher bitrate stability.")
        elif sw_diff < 0:
            lines.append(f"- Threshold made **{-sw_diff} fewer quality switches** in this run.")

        if acc_diff > 0:
            lines.append(f"- ML-Based matched the optimal quality **{acc_diff:.1f}% more often**.")
        else:
            lines.append(f"- Threshold matched optimal quality **{-acc_diff:.1f}% more often** in this run.")

        if qual_diff > 0:
            lines.append(f"- ML-Based delivered a higher average quality index (+{qual_diff:.2f} steps).")
        elif qual_diff < 0:
            lines.append(f"- Threshold delivered a slightly higher average quality index.")

        st.markdown("\n".join(lines))
        st.caption(
            "Results vary with random seed, duration, and model accuracy. "
            "Re-run with different seeds for statistical robustness."
        )
