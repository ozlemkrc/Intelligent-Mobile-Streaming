import numpy as np
import pandas as pd


class NetworkSimulator:
    """Generates synthetic mobile network performance data for three congestion scenarios."""

    FEATURES = [
        'throughput_dl', 'throughput_ul', 'latency',
        'packet_loss', 'jitter', 'signal_strength', 'mobility_speed'
    ]
    CLASSES = ['low', 'medium', 'high']

    # (mean, std, clip_min, clip_max) per feature per congestion class
    _PROFILES = {
        'low': {
            'throughput_dl':   (12.0, 2.5,  3.0,  20.0),   # Mbps
            'throughput_ul':   (5.0,  1.0,  1.0,  10.0),
            'latency':         (30.0, 8.0,  10.0, 60.0),    # ms
            'packet_loss':     (0.3,  0.2,  0.0,  1.2),     # %
            'jitter':          (2.5,  1.0,  0.5,  6.0),     # ms
            'signal_strength': (-65.0, 4.0, -75.0, -55.0),  # dBm
            'mobility_speed':  (15.0, 8.0,  0.0,  35.0),   # km/h
        },
        'medium': {
            'throughput_dl':   (5.0,  1.5,  1.5,  9.0),
            'throughput_ul':   (2.0,  0.7,  0.5,  4.0),
            'latency':         (90.0, 25.0, 40.0, 160.0),
            'packet_loss':     (3.0,  1.0,  0.5,  6.0),
            'jitter':          (12.0, 4.0,  4.0,  24.0),
            'signal_strength': (-77.0, 5.0, -88.0, -65.0),
            'mobility_speed':  (45.0, 12.0, 20.0, 70.0),
        },
        'high': {
            'throughput_dl':   (1.2,  0.6,  0.1,  3.5),
            'throughput_ul':   (0.4,  0.2,  0.05, 1.2),
            'latency':         (300.0, 80.0, 120.0, 600.0),
            'packet_loss':     (12.0, 4.0,  4.0,  25.0),
            'jitter':          (55.0, 20.0, 15.0, 120.0),
            'signal_strength': (-92.0, 5.0, -105.0, -80.0),
            'mobility_speed':  (85.0, 20.0, 50.0, 130.0),
        },
    }

    def generate_dataset(
        self,
        n_per_class: int = 400,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Return labeled DataFrame with n_per_class samples for each congestion level."""
        rng = np.random.default_rng(random_state)
        frames = []
        for label, profile in self._PROFILES.items():
            data = {}
            for feat in self.FEATURES:
                mean, std, lo, hi = profile[feat]
                samples = rng.normal(mean, std, n_per_class)
                data[feat] = np.clip(samples, lo, hi)
            df = pd.DataFrame(data)
            df['congestion'] = label
            frames.append(df)
        result = pd.concat(frames, ignore_index=True)
        # shuffle
        return result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def generate_time_series(
        self,
        duration: int = 300,
        random_state: int = 0
    ) -> pd.DataFrame:
        """
        Generate a duration-second time series that cycles through realistic
        congestion transitions (low → medium → high → medium → low → ...).
        """
        rng = np.random.default_rng(random_state)
        # Define a scenario sequence with segment lengths
        scenario_sequence = [
            ('low', 60), ('medium', 50), ('high', 40),
            ('medium', 40), ('low', 50), ('high', 30),
            ('medium', 30),
        ]
        # Tile until we cover 'duration'
        rows = []
        t = 0
        seg_idx = 0
        while t < duration:
            label, seg_len = scenario_sequence[seg_idx % len(scenario_sequence)]
            seg_len = min(seg_len, duration - t)
            profile = self._PROFILES[label]
            for _ in range(seg_len):
                row = {'time': t, 'congestion_true': label}
                for feat in self.FEATURES:
                    mean, std, lo, hi = profile[feat]
                    # Add per-step noise
                    val = rng.normal(mean, std * 0.4)
                    row[feat] = float(np.clip(val, lo, hi))
                rows.append(row)
                t += 1
            seg_idx += 1

        df = pd.DataFrame(rows)
        # Light temporal smoothing for realism
        for feat in self.FEATURES:
            df[feat] = df[feat].ewm(span=3).mean()
        return df.reset_index(drop=True)
