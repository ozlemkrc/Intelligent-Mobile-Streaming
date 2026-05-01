from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_LEVELS = {
    '240p':  300,    # kbps
    '480p':  1000,
    '720p':  2500,
    '1080p': 5000,
}
QUALITY_ORDER = ['240p', '480p', '720p', '1080p']

# Throughput thresholds (Mbps) that trigger each quality step up
THRESHOLDS = {
    '1080p': 4.5,
    '720p':  2.0,
    '480p':  0.8,
    # below 0.8 Mbps → 240p
}

# Congestion class → default quality (ML path)
CONGESTION_TO_QUALITY = {
    'low':    '1080p',
    'medium': '720p',
    'high':   '240p',
}

MAX_BUFFER = 30.0    # seconds
INIT_BUFFER = 5.0    # seconds
RESUME_AT   = 2.0    # seconds of buffer needed to exit rebuffering


def _threshold_select(throughput_mbps: float) -> str:
    if throughput_mbps >= THRESHOLDS['1080p']:
        return '1080p'
    if throughput_mbps >= THRESHOLDS['720p']:
        return '720p'
    if throughput_mbps >= THRESHOLDS['480p']:
        return '480p'
    return '240p'


def _optimal_quality(throughput_mbps: float) -> str:
    """Best quality that can be sustained without buffer drain."""
    throughput_kbps = throughput_mbps * 1000
    for q in reversed(QUALITY_ORDER):
        if throughput_kbps >= QUALITY_LEVELS[q]:
            return q
    return '240p'


class StreamingEngine:
    """Simulates an ABR streaming session over a network time series."""

    def simulate(
        self,
        ts: pd.DataFrame,
        method: str = 'threshold',
        predictions: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Run one streaming session.

        Parameters
        ----------
        ts          : time-series DataFrame from NetworkSimulator
        method      : 'threshold' or 'ml'
        predictions : array of congestion class strings (required when method='ml')

        Returns a per-second DataFrame with simulation state columns.
        """
        n = len(ts)
        records = []

        buffer = INIT_BUFFER
        current_quality = '480p'
        is_rebuffering = False
        rebuffer_events = 0
        rebuffer_seconds = 0
        quality_switches = 0
        last_switch_t = -1

        for t in range(n):
            row = ts.iloc[t]
            throughput_mbps = float(row['throughput_dl'])
            throughput_kbps = throughput_mbps * 1000

            # --- Select quality ---
            if method == 'threshold':
                target = _threshold_select(throughput_mbps)
            else:
                pred = predictions[t] if predictions is not None else 'medium'
                target = CONGESTION_TO_QUALITY.get(pred, '720p')

            if target != current_quality:
                quality_switches += 1
                last_switch_t = t
                current_quality = target

            bitrate_kbps = QUALITY_LEVELS[current_quality]
            optimal = _optimal_quality(throughput_mbps)

            # --- Buffer dynamics ---
            fill_rate = throughput_kbps / bitrate_kbps   # seconds of video per second

            if is_rebuffering:
                rebuffer_seconds += 1
                buffer = min(MAX_BUFFER, buffer + fill_rate)
                if buffer >= RESUME_AT:
                    is_rebuffering = False
            else:
                buffer = buffer + fill_rate - 1.0        # play 1s, download fill_rate s
                buffer = max(0.0, min(MAX_BUFFER, buffer))
                if buffer <= 0.0:
                    is_rebuffering = True
                    rebuffer_events += 1

            records.append({
                'time':            t,
                'throughput_mbps': throughput_mbps,
                'congestion_true': row.get('congestion_true', ''),
                'quality':         current_quality,
                'quality_kbps':    bitrate_kbps,
                'optimal_quality': optimal,
                'buffer_s':        round(buffer, 3),
                'rebuffering':     is_rebuffering,
                'rebuffer_events': rebuffer_events,
                'rebuffer_secs':   rebuffer_seconds,
                'quality_switches': quality_switches,
            })

        return pd.DataFrame(records)


def compute_metrics(sim: pd.DataFrame) -> dict:
    """Aggregate quality and stability metrics from a simulation DataFrame."""
    q_map = {q: i for i, q in enumerate(QUALITY_ORDER)}

    q_indices = sim['quality'].map(q_map)
    opt_indices = sim['optimal_quality'].map(q_map)

    # Fraction of time at each quality level
    q_dist = sim['quality'].value_counts(normalize=True).to_dict()

    # Mean quality index (higher = better)
    mean_q = q_indices.mean()

    # Adaptation accuracy: fraction of steps where selected == optimal
    adapt_acc = (sim['quality'] == sim['optimal_quality']).mean()

    # Quality mismatch: mean absolute difference in quality steps
    q_mismatch = (q_indices - opt_indices).abs().mean()

    return {
        'rebuffer_events': int(sim['rebuffer_events'].iloc[-1]),
        'rebuffer_secs': int(sim['rebuffer_secs'].iloc[-1]),
        'quality_switches': int(sim['quality_switches'].iloc[-1]),
        'mean_quality_idx': round(float(mean_q), 3),
        'adapt_accuracy': round(float(adapt_acc), 3),
        'quality_mismatch': round(float(q_mismatch), 3),
        'quality_distribution': {q: round(q_dist.get(q, 0), 3) for q in QUALITY_ORDER},
    }
