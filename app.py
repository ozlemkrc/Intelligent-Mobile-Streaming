"""
Intelligent Mobile Streaming — CSE 476 Term Project
Per-user ML-based adaptive bitrate streaming in mobile networks.

Tabs
----
  Overview        — system description and quick-start
  Persona Data    — per-user data generation and distribution explorer
  Model Training  — PersonalLSTM + KNN / RF baseline training on persona data
  Streaming       — ABR simulation: rule vs rate vs personal-ML
  Comparison      — metrics, multi-seed statistical analysis, win-rate table
  Cross-User      — controlled experiment proving personalisation value
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.persona_generator import PERSONAS, PersonaDataGenerator
from src.personal_model import PersonalTrainer, CrossUserEvaluator
from src.ml_classifier import FEATURES, CLASSES, NetworkClassifier
from src.streaming_engine import StreamingEngine, QUALITY_ORDER, compute_metrics
from src.performance_evaluator import (
    compare, plot_quality_timeline, plot_buffer, plot_estimator,
    plot_throughput, plot_comparison_bars, plot_quality_distribution,
    run_multi_seed, plot_multi_seed_bars, plot_multi_seed_distribution,
    compute_win_rates,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Mobile Streaming",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    height: 48px; padding: 0 16px;
    line-height: 48px; white-space: nowrap;
  }
  .stTabs [data-baseweb="tab"] > div { line-height: normal; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_STATE_KEYS = [
    'persona_data',         # Dict[persona_key, DataFrame]  — generated datasets
    'trainer',              # PersonalTrainer  — fitted on selected persona
    'baseline_clf',         # NetworkClassifier — KNN + RF trained on same data
    'ts',                   # streaming trace DataFrame
    'sim_rule', 'sim_th', 'sim_ml',   # ABR simulation results
    'ml_preds',             # LSTM predictions on ts
    'multi_seed_df',        # multi-seed comparison DataFrame
    'cross_user_results',   # CrossUserEvaluator output
]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 IMS Dashboard")
    st.divider()

    # ── 1. Data generation ────────────────────────────────────────────────────
    st.subheader("1 · Persona Data")
    persona_key = st.selectbox(
        "Active persona",
        list(PERSONAS.keys()),
        format_func=lambda k: PERSONAS[k].name,
    )
    n_sessions   = st.slider("Sessions per user", 5, 40, 15, 5)
    session_dur  = st.slider("Session duration (s)", 120, 600, 300, 60)
    data_seed    = st.number_input("Data seed", 0, 9999, 42, key="data_seed")

    if st.button("Generate All Persona Data", type="primary", use_container_width=True):
        gen = PersonaDataGenerator()
        with st.spinner("Generating per-user traces …"):
            st.session_state.persona_data = gen.generate_all_personas(
                n_sessions=n_sessions,
                session_duration=session_dur,
                seed=int(data_seed),
            )
        # Invalidate downstream state
        for k in ['trainer', 'baseline_clf', 'ts', 'sim_rule', 'sim_th',
                  'sim_ml', 'ml_preds', 'multi_seed_df', 'cross_user_results']:
            st.session_state[k] = None
        n_total = sum(len(d) for d in st.session_state.persona_data.values())
        st.success(f"{n_total:,} rows · {len(PERSONAS)} users")

    st.divider()

    # ── 2. Model training ─────────────────────────────────────────────────────
    st.subheader("2 · Train Models")
    lstm_window   = st.slider("LSTM window (steps)", 5, 30, 15, 5)
    lstm_epochs   = st.slider("LSTM epochs", 10, 60, 25, 5)
    lstm_patience = st.slider("Early-stop patience", 3, 15, 7, 1)
    knn_k         = st.slider("KNN — k", 1, 21, 5, 2)
    rf_trees      = st.slider("RF — trees", 50, 300, 100, 50)

    train_disabled = st.session_state.persona_data is None
    if st.button("Train on Selected Persona", type="primary",
                 use_container_width=True, disabled=train_disabled):
        df = st.session_state.persona_data[persona_key]

        trainer = PersonalTrainer(
            window=lstm_window, epochs=lstm_epochs, patience=lstm_patience
        )
        with st.spinner("Training personal LSTM …"):
            trainer.fit(df)
        st.session_state.trainer = trainer

        clf = NetworkClassifier(knn_k=knn_k, rf_trees=rf_trees)
        with st.spinner("Training KNN & Random Forest …"):
            clf.train(df)
        st.session_state.baseline_clf = clf

        for k in ['ts', 'sim_rule', 'sim_th', 'sim_ml', 'ml_preds', 'multi_seed_df']:
            st.session_state[k] = None
        st.success("Models trained.")

    st.divider()

    # ── 3. Streaming simulation ───────────────────────────────────────────────
    st.subheader("3 · Streaming")
    sim_hour = st.slider("Stream starts at hour", 0.0, 23.0, 8.5, 0.5)
    sim_dur  = st.slider("Duration (s)", 60, 600, 300, 30)
    sim_seed = st.number_input("Sim seed", 0, 9999, 7, key="sim_seed")

    sim_disabled = st.session_state.trainer is None
    if st.button("Run Simulation", type="primary", use_container_width=True,
                 disabled=sim_disabled):
        gen = PersonaDataGenerator()
        ts  = gen.generate_streaming_trace(
            persona_key, duration=sim_dur,
            hour_of_day=sim_hour, seed=int(sim_seed),
        )
        st.session_state.ts = ts

        preds = st.session_state.trainer.predict_series(ts)
        st.session_state.ml_preds = preds

        engine = StreamingEngine()
        st.session_state.sim_rule = engine.simulate(ts, method='rule')
        st.session_state.sim_th   = engine.simulate(ts, method='threshold')
        st.session_state.sim_ml   = engine.simulate(ts, method='ml', predictions=preds)
        st.session_state.multi_seed_df = None
        st.success("Simulation complete.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Persona Data",
    "🤖 Model Training",
    "▶️ Streaming",
    "📈 Comparison",
    "🧪 Cross-User",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.title("Intelligent Mobile Streaming")
    st.markdown("""
    ### System overview

    | Step | Module | Description |
    |------|--------|-------------|
    | 1 | **Persona Generator** | Five user archetypes, each with home / work / commute anchors. Metrics emerge from a capacity model coupling RSSI and cell load — labels are not pre-assigned |
    | 2 | **Personal LSTM** | PyTorch 2-layer LSTM trained on *one user's* data. Predicts video quality level (240p–1080p) directly using look-ahead labels derived from near-future throughput |
    | 3 | **KNN / RF baselines** | Scikit-learn classifiers trained on the same persona data for fair comparison |
    | 4 | **Streaming Engine** | Buffer-based ABR simulation comparing rule-based, rate-based, and ML-based controllers |
    | 5 | **Cross-User Experiment** | Proves personalisation: personal model vs generic model on each user's held-out test data |

    ### Quick start
    1. **Generate All Persona Data** in the sidebar
    2. **Train on Selected Persona** — LSTM + KNN + RF all fitted on the same data
    3. **Run Simulation** — see quality timelines in *Streaming*
    4. Switch to **Comparison** for multi-seed statistical analysis
    5. Use **Cross-User** to run the personalisation proof experiment

    ### The five personas

    | Persona | Typical pattern |
    |---------|-----------------|
    | Urban Commuter (Alice) | Bimodal — excellent at office/home, terrible in subway |
    | Suburban Student (Bob) | Mostly low congestion (campus 5G), medium at home |
    | Rural Remote (Charlie) | Consistently weak signal → medium–high congestion |
    | Dense Urban (Dana)     | Strong signal, high cell load → persistent medium |
    | Frequent Traveler (Eve)| Highly variable; frequent handoffs at speed |

    ### Network parameters generated

    `throughput_dl` · `throughput_ul` · `latency` · `packet_loss` · `jitter` · `signal_strength` · `mobility_speed`
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Persona Data
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Persona Data Explorer")

    if st.session_state.persona_data is None:
        st.info("Generate data from the sidebar first.")
    else:
        pds = st.session_state.persona_data

        # Congestion class breakdown
        st.subheader("Congestion Class Breakdown per Persona")
        class_rows = []
        for k, df in pds.items():
            cnt = df['congestion_true'].value_counts(normalize=True)
            class_rows.append({
                'Persona':    PERSONAS[k].name,
                'Low (%)':    round(cnt.get('low',    0) * 100, 1),
                'Medium (%)': round(cnt.get('medium', 0) * 100, 1),
                'High (%)':   round(cnt.get('high',   0) * 100, 1),
                'Rows':       len(df),
            })
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Distributions differ meaningfully across personas because labels emerge "
            "from the physics of each user's locations — not from pre-set class buckets."
        )

        st.divider()

        # Feature distribution comparison
        st.subheader("Feature Distributions by Persona")
        feat_sel = st.selectbox(
            "Feature", FEATURES,
            format_func=lambda f: f.replace('_', ' ').title(),
        )
        combined = pd.concat(
            [df.assign(persona_name=PERSONAS[k].name) for k, df in pds.items()],
            ignore_index=True,
        )
        fig_dist = px.histogram(
            combined, x=feat_sel, color='persona_name', barmode='overlay',
            opacity=0.65, nbins=60,
            labels={feat_sel: feat_sel.replace('_', ' ').title()},
            title=f"{feat_sel.replace('_', ' ').title()} — all personas overlaid",
        )
        fig_dist.update_layout(height=360, legend_title='Persona')
        st.plotly_chart(fig_dist, use_container_width=True)

        # Selected persona detail
        st.subheader(f"Selected Persona — {PERSONAS[persona_key].name}")
        df_sel = pds[persona_key]
        col_loc, col_cls = st.columns(2)
        with col_loc:
            loc_counts = df_sel['location'].value_counts()
            fig_loc = px.pie(values=loc_counts.values, names=loc_counts.index,
                             title='Time at each location')
            fig_loc.update_layout(height=300)
            st.plotly_chart(fig_loc, use_container_width=True)
        with col_cls:
            cls_counts = df_sel['congestion_true'].value_counts()
            fig_cls = px.pie(
                values=cls_counts.values, names=cls_counts.index,
                color=cls_counts.index,
                color_discrete_map={'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'},
                title='Congestion class distribution',
            )
            fig_cls.update_layout(height=300)
            st.plotly_chart(fig_cls, use_container_width=True)

        st.subheader("Pairwise Scatter (500-row sample)")
        pair_feats = st.multiselect(
            "Features",
            FEATURES,
            default=['throughput_dl', 'latency', 'signal_strength', 'packet_loss'],
        )
        if len(pair_feats) >= 2:
            sample = df_sel.sample(min(500, len(df_sel)), random_state=1)
            fig_pair = px.scatter_matrix(
                sample, dimensions=pair_feats, color='congestion_true',
                color_discrete_map={'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'},
            )
            fig_pair.update_traces(diagonal_visible=False, marker_size=3)
            fig_pair.update_layout(height=560)
            st.plotly_chart(fig_pair, use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_sel.head(200), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Training
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Model Training")

    if st.session_state.trainer is None:
        st.info("Train models from the sidebar first.")
    else:
        trainer: PersonalTrainer = st.session_state.trainer
        clf: NetworkClassifier   = st.session_state.baseline_clf
        df_train = st.session_state.persona_data[persona_key]

        st.subheader(f"Training data — {PERSONAS[persona_key].name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{len(df_train):,}")
        c2.metric("LSTM window", trainer.window)
        c3.metric("LSTM epochs trained", len(trainer.train_losses))

        # LSTM training curves
        if trainer.train_losses:
            st.subheader("LSTM Training Curves")
            curve_df = pd.DataFrame({
                'epoch':      list(range(1, len(trainer.train_losses) + 1)),
                'Train loss': trainer.train_losses,
                'Val loss':   trainer.val_losses,
            })
            fig_loss = px.line(
                curve_df.melt('epoch', var_name='Split', value_name='Loss'),
                x='epoch', y='Loss', color='Split',
                title='Cross-entropy loss per epoch',
            )
            fig_loss.update_layout(height=300)
            st.plotly_chart(fig_loss, use_container_width=True)

        # LSTM evaluation on full dataset
        st.subheader("LSTM — Quality Prediction on Full Persona Dataset")
        lstm_metrics = trainer.evaluate(df_train)
        lc1, lc2 = st.columns(2)
        lc1.metric("Accuracy", f"{lstm_metrics['accuracy']:.3f}")
        lc2.metric("F1 Macro", f"{lstm_metrics['f1_macro']:.3f}")

        cm_lstm = lstm_metrics['confusion_matrix']
        cls_names = lstm_metrics['class_names']
        fig_cm_lstm = px.imshow(
            cm_lstm, text_auto=True,
            x=cls_names, y=cls_names,
            color_continuous_scale='Blues',
            labels={'x': 'Predicted', 'y': 'Actual'},
            title='LSTM Confusion Matrix',
        )
        fig_cm_lstm.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_cm_lstm, use_container_width=True)

        # KNN / RF baselines
        st.divider()
        st.subheader("KNN & Random Forest — Baselines")
        summary_rows = []
        for name, r in clf.results.items():
            rep = r['report']
            # label_names order matches LabelEncoder (alphabetical: 1080p/240p/480p/720p)
            summary_rows.append({
                'Model':    name,
                'Accuracy': f"{r['accuracy']:.3f}",
                'F1 Macro': f"{r['f1_macro']:.3f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        cm_cols = st.columns(len(clf.results))
        for col, (name, r) in zip(cm_cols, clf.results.items()):
            fig_cm = px.imshow(
                r['confusion_matrix'], text_auto=True,
                color_continuous_scale='Blues',
                labels={'x': 'Predicted', 'y': 'Actual'},
                title=name,
            )
            fig_cm.update_layout(height=280, coloraxis_showscale=False)
            col.plotly_chart(fig_cm, use_container_width=True)

        # RF feature importance
        st.subheader("Random Forest — Feature Importance")
        fi = clf.feature_importances_
        fig_fi = px.bar(
            x=list(fi.values()), y=list(fi.keys()),
            orientation='h',
            color=list(fi.values()), color_continuous_scale='Teal',
            labels={'x': 'Importance', 'y': 'Feature'},
        )
        fig_fi.update_layout(height=300, coloraxis_showscale=False,
                             yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Streaming
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Adaptive Bitrate Streaming Simulation")

    if st.session_state.sim_th is None:
        st.info("Run a simulation from the sidebar first.")
    else:
        ts       = st.session_state.ts
        sim_rule = st.session_state.sim_rule
        sim_th   = st.session_state.sim_th
        sim_ml   = st.session_state.sim_ml
        m_r      = compute_metrics(sim_rule)
        m_t      = compute_metrics(sim_th)
        m_m      = compute_metrics(sim_ml)

        st.subheader("QoE Score  *(avg bitrate − α·rebuffer_secs − β·switches)*")
        st.caption("α = 0.3 Mbps/s (stall penalty)  ·  β = 0.02 Mbps/switch (smoothness).  Higher is better.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule-Based QoE",    f"{m_r['qoe_score']:.3f}")
        c2.metric("Rate-Based QoE",    f"{m_t['qoe_score']:.3f}",
                  delta=round(m_t['qoe_score'] - m_r['qoe_score'], 3))
        c3.metric("Personal-ML QoE",   f"{m_m['qoe_score']:.3f}",
                  delta=round(m_m['qoe_score'] - m_r['qoe_score'], 3))

        st.divider()
        col_r, col_t, col_m = st.columns(3)
        for col, label, m in [
            (col_r, "Rule-Based  *(naive threshold)*",           m_r),
            (col_t, "Rate-Based  *(harmonic-mean estimate)*",    m_t),
            (col_m, "Personal-ML  *(LSTM quality prediction)*",  m_m),
        ]:
            with col:
                st.markdown(f"#### {label}")
                st.metric("Avg Bitrate",         f"{m['avg_bitrate_mbps']:.3f} Mbps")
                st.metric("Rebuffering Events",  m['rebuffer_events'])
                st.metric("Rebuffering (s)",     m['rebuffer_secs'])
                st.metric("Quality Switches",    m['quality_switches'])
                st.metric("Adaptation Accuracy", f"{m['adapt_accuracy']*100:.1f}%")

        st.divider()
        st.plotly_chart(plot_throughput(ts),                          use_container_width=True)
        st.plotly_chart(plot_estimator(sim_th, sim_ml, sim_rule),     use_container_width=True)
        st.plotly_chart(plot_quality_timeline(sim_th, sim_ml, sim_rule), use_container_width=True)
        st.plotly_chart(plot_buffer(sim_th, sim_ml, sim_rule),        use_container_width=True)

        with st.expander("Raw simulation data"):
            view = st.radio("Show:", ['Rule-Based', 'Rate-Based', 'ML-Based'], horizontal=True)
            st.dataframe({'Rule-Based': sim_rule, 'Rate-Based': sim_th,
                          'ML-Based': sim_ml}[view], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Performance Comparison")

    if st.session_state.sim_th is None:
        st.info("Run a simulation from the sidebar first.")
    else:
        sim_rule = st.session_state.sim_rule
        sim_th   = st.session_state.sim_th
        sim_ml   = st.session_state.sim_ml
        metrics  = compare(sim_th, sim_ml, sim_rule)

        st.plotly_chart(plot_comparison_bars(metrics),      use_container_width=True)
        st.plotly_chart(plot_quality_distribution(metrics), use_container_width=True)

        st.subheader("Metric Table")
        metric_rows = []
        for key, label in [
            ('qoe_score',        'QoE Score (Mbps-eq.) ↑'),
            ('avg_bitrate_mbps', 'Avg Bitrate (Mbps) ↑'),
            ('rebuffer_events',  'Rebuffering Events ↓'),
            ('rebuffer_secs',    'Rebuffering Duration (s) ↓'),
            ('quality_switches', 'Quality Switches ↓'),
            ('mean_quality_idx', 'Mean Quality Index ↑'),
            ('adapt_accuracy',   'Adaptation Accuracy ↑'),
            ('quality_mismatch', 'Quality Mismatch ↓'),
        ]:
            metric_rows.append({
                'Metric':     label,
                'Rule-Based': metrics['Rule-Based'][key],
                'Rate-Based': metrics['Threshold'][key],
                'ML-Based':   metrics['ML-Based'][key],
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # ── Multi-seed statistical analysis ────────────────────────────────────
        st.divider()
        st.subheader("Statistical Robustness — Multi-Seed Analysis")
        st.markdown(
            "Run **N independent simulations** at different hours of the day and "
            "compare mean ± std and win-rates per metric across methods."
        )
        col_n, col_d = st.columns(2)
        with col_n:
            n_runs     = st.slider("Number of runs", 5, 50, 20, 5, key="ms_n")
        with col_d:
            dur_runs   = st.slider("Duration per run (s)", 60, 300, 180, 30, key="ms_dur")

        if st.button("Run Multi-Seed Analysis", type="primary"):
            prog = st.progress(0.0, text="Running …")
            df_runs = run_multi_seed(
                st.session_state.trainer,
                persona_key=persona_key,
                n_runs=n_runs,
                duration=dur_runs,
                progress_callback=lambda p: prog.progress(p, text=f"Running … {int(p*100)} %"),
            )
            prog.empty()
            st.session_state.multi_seed_df = df_runs

        if st.session_state.multi_seed_df is not None:
            df_runs = st.session_state.multi_seed_df
            st.plotly_chart(plot_multi_seed_bars(df_runs), use_container_width=True)

            baseline = st.radio(
                "Win-rate baseline:", ['Rule-Based', 'Threshold'], horizontal=True,
            )
            st.dataframe(compute_win_rates(df_runs, baseline=baseline),
                         use_container_width=True, hide_index=True)

            box_m = st.selectbox(
                "Distribution view:",
                ['qoe_score', 'avg_bitrate_mbps', 'rebuffer_events', 'rebuffer_secs',
                 'quality_switches', 'mean_quality_idx', 'adapt_accuracy'],
            )
            st.plotly_chart(plot_multi_seed_distribution(df_runs, box_m),
                            use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Cross-User Experiment
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Cross-User Personalisation Experiment")
    st.markdown("""
    For each user: train a **personal model** on their data only, and a **generic model**
    on all *other* users' data.  Both are evaluated on the same held-out test set (last 20 %).

    If the personal model consistently outperforms the generic model, the claim is proven:
    the LSTM has learned user-specific patterns that a population-level model misses.
    """)

    if st.session_state.persona_data is None:
        st.info("Generate persona data from the sidebar first.")
    else:
        col_ep, col_win, col_pat = st.columns(3)
        with col_ep:
            xp_epochs  = st.slider("LSTM epochs", 10, 60, 25, 5, key="xp_ep")
        with col_win:
            xp_window  = st.slider("Window (steps)", 5, 30, 15, 5, key="xp_win")
        with col_pat:
            xp_patience = st.slider("Early-stop patience", 3, 15, 7, 1, key="xp_pat")

        if st.button("Run Cross-User Experiment", type="primary"):
            prog = st.progress(0.0, text="Training models …")
            ev = CrossUserEvaluator(
                test_frac=0.20,
                window=xp_window,
                epochs=xp_epochs,
                patience=xp_patience,
            )
            res = ev.run(
                st.session_state.persona_data,
                progress_callback=lambda p: prog.progress(
                    p, text=f"Training models … {int(p*100)} %"
                ),
            )
            prog.empty()
            st.session_state.cross_user_results = res
            st.success("Experiment complete.")

        if st.session_state.cross_user_results is not None:
            res = st.session_state.cross_user_results

            st.subheader("Results — Personal vs Generic Accuracy")
            display = res[[
                'persona_name', 'personal_acc', 'generic_acc', 'delta_acc',
                'personal_f1',  'generic_f1',  'delta_f1',
            ]].rename(columns={
                'persona_name': 'Persona',
                'personal_acc': 'Personal Acc',
                'generic_acc':  'Generic Acc',
                'delta_acc':    'Δ Acc',
                'personal_f1':  'Personal F1',
                'generic_f1':   'Generic F1',
                'delta_f1':     'Δ F1',
            })
            st.dataframe(
                display.style.background_gradient(
                    subset=['Δ Acc', 'Δ F1'],
                    cmap='RdYlGn', vmin=-0.05, vmax=0.20,
                ),
                use_container_width=True, hide_index=True,
            )

            mean_d_acc = res['delta_acc'].mean()
            wins       = (res['delta_acc'] > 0).sum()
            st.markdown(
                f"**Mean Δ accuracy:** `{mean_d_acc:+.4f}`  |  "
                f"**Personal wins:** `{wins}/{len(res)}` personas"
            )

            fig_bar = px.bar(
                res, x='persona_name',
                y=['personal_acc', 'generic_acc'],
                barmode='group',
                color_discrete_map={
                    'personal_acc': '#a6e3a1',
                    'generic_acc':  '#f38ba8',
                },
                labels={'value': 'Accuracy', 'persona_name': 'Persona',
                        'variable': 'Model'},
                title='Personal vs Generic Accuracy per User',
            )
            fig_bar.for_each_trace(lambda t: t.update(
                name='Personal' if t.name == 'personal_acc' else 'Generic'
            ))
            fig_bar.update_layout(height=380, xaxis_tickangle=-15,
                                  legend_title='Model')
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("Raw results"):
                st.dataframe(res, use_container_width=True, hide_index=True)
