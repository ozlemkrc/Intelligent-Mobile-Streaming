"""
benchmark_numbers.py

Runs the full pipeline and prints averaged numbers for PRESENTATION.md.
Usage:  python benchmark_numbers.py
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.persona_generator import PersonaDataGenerator, PERSONAS
from src.personal_model import PersonalTrainer, CrossUserEvaluator
from src.ml_classifier import NetworkClassifier
from src.streaming_engine import StreamingEngine, compute_metrics
from src.performance_evaluator import run_multi_seed, compute_win_rates

PERSONA_KEYS = list(PERSONAS.keys())
N_SESSIONS   = 20
DURATION     = 300
N_SEEDS      = 20
BASE_SEED    = 1000

print("=" * 62)
print("BENCHMARK — Intelligent Mobile Streaming")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────
# 1. Generate data for every persona
# ─────────────────────────────────────────────────────────────────
print("\n[1/4] Generating data ...")
gen = PersonaDataGenerator()
persona_dfs: dict[str, pd.DataFrame] = {}
for key in PERSONA_KEYS:
    sessions = []
    for s in range(N_SESSIONS):
        hour = float((BASE_SEED + s) % 24)
        df   = gen.generate_streaming_trace(
            key, duration=DURATION, hour_of_day=hour, seed=BASE_SEED + s
        )
        sessions.append(df)
    persona_dfs[key] = pd.concat(sessions, ignore_index=True)
    print(f"  {key}: {len(persona_dfs[key])} rows")

all_df = pd.concat(persona_dfs.values(), ignore_index=True)

# ─────────────────────────────────────────────────────────────────
# 2. KNN / RF classification accuracy (trained on all personas)
# ─────────────────────────────────────────────────────────────────
print("\n[2/4] Training KNN & Random Forest ...")
clf = NetworkClassifier(knn_k=5, rf_trees=100, random_state=42)
results = clf.train(all_df, test_size=0.2)

print(f"\n  KNN  accuracy={results['KNN']['accuracy']:.4f}  "
      f"f1_macro={results['KNN']['f1_macro']:.4f}")
print(f"  RF   accuracy={results['Random Forest']['accuracy']:.4f}  "
      f"f1_macro={results['Random Forest']['f1_macro']:.4f}")

# Per-persona LSTM accuracy
print("\n[2b] Training per-persona LSTM ...")
trainers: dict[str, PersonalTrainer] = {}
lstm_accs: list[float] = []
lstm_f1s:  list[float] = []
for key in PERSONA_KEYS:
    t = PersonalTrainer(epochs=30, patience=7, horizon=5)
    stats = t.fit(persona_dfs[key])
    trainers[key] = t
    # Evaluate on the validation portion (fit() already reports it)
    acc = stats['val_accuracy']
    f1  = stats['val_f1']
    lstm_accs.append(acc)
    lstm_f1s.append(f1)
    print(f"  {key}: acc={acc:.4f}  f1={f1:.4f}  "
          f"epochs={stats['epochs_trained']}")

print(f"\n  LSTM Personal (mean): acc={np.mean(lstm_accs):.4f}  "
      f"f1={np.mean(lstm_f1s):.4f}")

# ─────────────────────────────────────────────────────────────────
# 3. Multi-seed streaming (all personas, averaged)
# ─────────────────────────────────────────────────────────────────
print("\n[3/4] Multi-seed streaming simulation ...")
all_runs_list: list[pd.DataFrame] = []

for key in PERSONA_KEYS:
    print(f"  {key} ...", end=' ', flush=True)
    runs = run_multi_seed(
        trainers[key],
        persona_key=key,
        n_runs=N_SEEDS,
        duration=DURATION,
        base_seed=BASE_SEED,
    )
    all_runs_list.append(runs)
    print("done")

all_runs = pd.concat(all_runs_list, ignore_index=True)

# Aggregate per method across all personas × seeds
metrics_to_show = [
    'qoe_score', 'avg_bitrate_mbps',
    'rebuffer_events', 'rebuffer_secs', 'quality_switches',
]
print("\n  Mean ± Std across all personas × seeds:")
print(f"  {'Metric':<22} {'Rule-Based':>18} {'Rate-Based':>18} {'ML-Based':>18}")
for m in metrics_to_show:
    row = []
    for method in ['Rule-Based', 'Threshold', 'ML-Based']:
        sub = all_runs[all_runs['method'] == method][m]
        row.append(f"{sub.mean():.3f} ± {sub.std():.3f}")
    print(f"  {m:<22} {row[0]:>18} {row[1]:>18} {row[2]:>18}")

# Win rates vs Rate-Based (Threshold)
win_df = compute_win_rates(all_runs, baseline='Threshold')
print("\n  Win rates (ML-Based vs Rate-Based):")
for _, r in win_df.iterrows():
    print(f"  {r['Metric']:<22} ML win rate = {r['ML win rate']}")

# ─────────────────────────────────────────────────────────────────
# 4. Cross-user personalization experiment
# ─────────────────────────────────────────────────────────────────
print("\n[4/4] Cross-user personalization experiment ...")
evaluator = CrossUserEvaluator(test_frac=0.20, epochs=30, patience=7, horizon=5)
cross_df  = evaluator.run(persona_dfs)

print("\n  Cross-user results:")
print(cross_df.to_string(index=False))
print(f"\n  Mean personal_acc = {cross_df['personal_acc'].mean():.4f}")
print(f"  Mean generic_acc  = {cross_df['generic_acc'].mean():.4f}")
print(f"  Mean delta_acc    = {cross_df['delta_acc'].mean():.4f}")
print(f"  Mean personal_f1  = {cross_df['personal_f1'].mean():.4f}")
print(f"  Mean generic_f1   = {cross_df['generic_f1'].mean():.4f}")
print(f"  Mean delta_f1     = {cross_df['delta_f1'].mean():.4f}")

print("\n" + "=" * 62)
print("DONE — copy numbers above into PRESENTATION.md")
print("=" * 62)
