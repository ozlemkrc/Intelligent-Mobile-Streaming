from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

QUALITY_LEVELS = {
    '240p':  300,    # kbps
    '480p':  1000,
    '720p':  2500,
    '1080p': 5000,
}
QUALITY_ORDER = ['240p', '480p', '720p', '1080p']

# Sliding window for the throughput estimator (seconds).
DEFAULT_ESTIMATOR_WINDOW = 5

# Standard QoE weights (Mbps-equivalent units).
# Formula: QoE = avg_bitrate_mbps − α·rebuffer_secs − β·quality_switches
# α: each second of stall costs the equivalent of 0.3 Mbps of average quality.
# β: each quality switch costs 0.02 Mbps (smoothness penalty).
QOE_ALPHA = 0.3
QOE_BETA  = 0.02

# Naive threshold rule (proposal's "traditional threshold-based" baseline).
# Maps the most recent observed throughput directly to a quality bucket
# without any smoothing, estimation, or learned signal. This is the
# textbook rule-based ABR controller the ML method is meant to beat.
RULE_THRESHOLDS = [
    (0.5,  '240p'),
    (1.5,  '480p'),
    (3.5,  '720p'),
]


def _rule_quality(last_throughput_mbps: float) -> str:
    for thresh, q in RULE_THRESHOLDS:
        if last_throughput_mbps < thresh:
            return q
    return '1080p'

# Buffer model
MAX_BUFFER  = 30.0
INIT_BUFFER = 5.0
RESUME_AT   = 2.0


class ThroughputEstimator:
    """Harmonic mean over a sliding window — standard ABR practice
    (dash.js, Pensieve baseline). Harmonic mean is conservative under
    variance: a single low sample pulls the estimate down sharply."""

    def __init__(self, window: int = DEFAULT_ESTIMATOR_WINDOW):
        self.window = window
        self.history: deque = deque(maxlen=window)

    def observe(self, throughput_mbps: float) -> None:
        self.history.append(max(throughput_mbps, 0.01))

    def estimate(self) -> float:
        if not self.history:
            return 1.0  # cold-start fallback
        return len(self.history) / sum(1.0 / x for x in self.history)


def _select_quality_under(effective_mbps: float) -> str:
    """Highest quality whose bitrate fits under the given effective throughput."""
    effective_kbps = effective_mbps * 1000
    for q in reversed(QUALITY_ORDER):
        if QUALITY_LEVELS[q] <= effective_kbps:
            return q
    return '240p'


def _optimal_quality(throughput_mbps: float) -> str:
    """Oracle: best quality with perfect knowledge of the current throughput."""
    return _select_quality_under(throughput_mbps)


class StreamingEngine:
    """Simulates an ABR streaming session over a network time series.

    Three controllers:
    - rule      : naive threshold on the most recent throughput observation
    - threshold : harmonic-mean estimate over a sliding window
    - ml        : LSTM-predicted quality level used directly (no safety scaling)
    """

    def simulate(
        self,
        ts: pd.DataFrame,
        method: str = 'threshold',
        predictions: np.ndarray | None = None,
        estimator_window: int = DEFAULT_ESTIMATOR_WINDOW,
    ) -> pd.DataFrame:

        n = len(ts)
        records = []
        estimator = ThroughputEstimator(window=estimator_window)

        buffer = INIT_BUFFER
        current_quality = '240p'   # safe cold start
        is_rebuffering = False
        rebuffer_events = 0
        rebuffer_seconds = 0
        quality_switches = 0
        last_throughput = 1.0  # rule-method cold-start observation

        for t in range(n):
            row = ts.iloc[t]
            throughput_mbps = float(row['throughput_dl'])
            throughput_kbps = throughput_mbps * 1000

            # ── Quality decision (uses PRIOR observations only) ──
            estimate = estimator.estimate()
            if method == 'rule':
                # Naive threshold rule: pick quality bucket directly from
                # the most recent observed throughput. No smoothing.
                effective = last_throughput
                pred_label = ''
                target = _rule_quality(last_throughput)
            elif method == 'threshold':
                effective = estimate
                pred_label = ''
                target = _select_quality_under(effective)
            else:
                # LSTM predicts quality level directly — use it as-is.
                target = predictions[t] if predictions is not None else '480p'
                if target not in QUALITY_LEVELS:
                    target = '480p'
                pred_label = target
                effective  = QUALITY_LEVELS[target] / 1000.0

            if target != current_quality:
                quality_switches += 1
                current_quality = target

            bitrate_kbps = QUALITY_LEVELS[current_quality]
            optimal = _optimal_quality(throughput_mbps)

            # ── Buffer dynamics (uses ACTUAL throughput) ──
            fill_rate = throughput_kbps / bitrate_kbps

            if is_rebuffering:
                rebuffer_seconds += 1
                buffer = min(MAX_BUFFER, buffer + fill_rate)
                if buffer >= RESUME_AT:
                    is_rebuffering = False
            else:
                buffer = buffer + fill_rate - 1.0
                buffer = max(0.0, min(MAX_BUFFER, buffer))
                if buffer <= 0.0:
                    is_rebuffering = True
                    rebuffer_events += 1

            # Observe AFTER the decision (causal: the controller can only
            # use throughput it has already seen).
            estimator.observe(throughput_mbps)
            last_throughput = throughput_mbps

            records.append({
                'time':              t,
                'throughput_mbps':   throughput_mbps,
                'throughput_est':    round(estimate, 3),
                'effective_mbps':    round(effective, 3),
                'congestion_true':   row.get('congestion_true', ''),
                'predicted_class':   pred_label,
                'quality':           current_quality,
                'quality_kbps':      bitrate_kbps,
                'optimal_quality':   optimal,
                'buffer_s':          round(buffer, 3),
                'rebuffering':       is_rebuffering,
                'rebuffer_events':   rebuffer_events,
                'rebuffer_secs':     rebuffer_seconds,
                'quality_switches':  quality_switches,
            })

        return pd.DataFrame(records)


def compute_metrics(sim: pd.DataFrame) -> dict:
    """Aggregate quality and stability metrics from a simulation DataFrame."""
    q_map = {q: i for i, q in enumerate(QUALITY_ORDER)}
    q_indices = sim['quality'].map(q_map)
    opt_indices = sim['optimal_quality'].map(q_map)
    q_dist = sim['quality'].value_counts(normalize=True).to_dict()
    avg_bitrate_mbps = round(float(sim['quality_kbps'].mean()) / 1000, 3)
    rebuf_secs = int(sim['rebuffer_secs'].iloc[-1])
    switches   = int(sim['quality_switches'].iloc[-1])
    qoe = round(avg_bitrate_mbps - QOE_ALPHA * rebuf_secs - QOE_BETA * switches, 3)
    return {
        'qoe_score':            qoe,
        'avg_bitrate_mbps':     avg_bitrate_mbps,
        'rebuffer_events':      int(sim['rebuffer_events'].iloc[-1]),
        'rebuffer_secs':        rebuf_secs,
        'quality_switches':     switches,
        'mean_quality_idx':     round(float(q_indices.mean()), 3),
        'adapt_accuracy':       round(float((sim['quality'] == sim['optimal_quality']).mean()), 3),
        'quality_mismatch':     round(float((q_indices - opt_indices).abs().mean()), 3),
        'quality_distribution': {q: round(q_dist.get(q, 0), 3) for q in QUALITY_ORDER},
    }
