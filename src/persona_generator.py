"""
Per-user mobile network trace generator.

Instead of sampling from fixed class distributions, each user persona is
modelled as having three spatial anchors (home, work, commute corridor)
with distinct RF and cell-load characteristics.  Network metrics emerge
from a physically-motivated capacity model:

  throughput ∝ SINR_efficiency(RSSI) × (1 − cell_load)^α
  latency     ∝ base_latency × (1 + queuing_factor × cell_load²)
  packet_loss ← poor SINR + high load, both compounding

Congestion labels are derived from the physics rather than pre-assigned,
so class boundaries are realistically overlapping and user-specific.

The five personas cover a spectrum of urban, suburban, and rural
conditions.  When a model trained on one persona's data is evaluated on
another's, it degrades measurably — which is the empirical basis for
the claim that personalized models outperform generic ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

FEATURES = [
    'throughput_dl', 'throughput_ul', 'latency',
    'packet_loss', 'jitter', 'signal_strength', 'mobility_speed',
]
CLASSES = ['low', 'medium', 'high']

# ---------------------------------------------------------------------------
# Empirical urban cell-load profile over 24 hours (index = integer hour).
# Based on typical 4G/5G measurement studies of urban macro cells.
# ---------------------------------------------------------------------------
_LOAD_CURVE = np.array([
    0.10, 0.08, 0.07, 0.07, 0.08, 0.14,   # 00-05  night / early morning
    0.28, 0.58, 0.82, 0.72, 0.62, 0.64,   # 06-11  morning ramp / rush
    0.76, 0.66, 0.60, 0.63, 0.70, 0.88,   # 12-17  afternoon / evening build
    0.92, 0.82, 0.72, 0.56, 0.36, 0.18,   # 18-23  streaming peak / night
])

# AR(1) smoothing coefficient: 0 = no memory, 1 = frozen signal
_AR1_ALPHA = 0.72

# RSSI reference point for the logistic SINR model
_RSSI_THRESHOLD = -80.0   # dBm  — midpoint where SINR factor = 0.5
_RSSI_SCALE     =   8.0   # dBm  — steepness of the logistic


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocationProfile:
    """RF and capacity characteristics at a physical anchor location."""
    name: str
    rssi_mean: float        # dBm;  e.g. −62 (good 5G) … −92 (rural edge)
    rssi_shadow_std: float  # log-normal shadowing spread, typically 4–10 dBm
    cell_load_base: float   # baseline cell utilisation [0, 1]
    peak_dl_mbps: float     # theoretical peak downlink under ideal conditions
    peak_ul_mbps: float
    latency_min_ms: float   # RTT at near-zero load
    latency_max_ms: float   # RTT at full load (queuing-dominated)
    speed_mean: float       # typical device speed (km/h)
    speed_std: float
    is_mobile: bool = False


@dataclass
class PersonaProfile:
    """A user archetype with three location anchors and a daily schedule."""
    name: str               # human-readable label shown in the UI
    locations: Dict[str, LocationProfile]
    # Each entry: (hour_start, hour_end, location_key) — fractional hours OK
    schedule_weekday: List[Tuple[float, float, str]]
    schedule_weekend: List[Tuple[float, float, str]]


# ---------------------------------------------------------------------------
# The five user personas
# ---------------------------------------------------------------------------

PERSONAS: Dict[str, PersonaProfile] = {

    'urban_commuter': PersonaProfile(
        name='Urban Commuter (Alice)',
        locations={
            'home': LocationProfile(
                'Suburban LTE', rssi_mean=-70, rssi_shadow_std=5,
                cell_load_base=0.35, peak_dl_mbps=50, peak_ul_mbps=20,
                latency_min_ms=25, latency_max_ms=120,
                speed_mean=0, speed_std=1,
            ),
            'work': LocationProfile(
                'Downtown 5G Office', rssi_mean=-62, rssi_shadow_std=4,
                cell_load_base=0.55, peak_dl_mbps=200, peak_ul_mbps=80,
                latency_min_ms=15, latency_max_ms=80,
                speed_mean=0, speed_std=1,
            ),
            'commute': LocationProfile(
                'Subway / Bus Corridor', rssi_mean=-90, rssi_shadow_std=8,
                cell_load_base=0.80, peak_dl_mbps=30, peak_ul_mbps=10,
                latency_min_ms=60, latency_max_ms=400,
                speed_mean=40, speed_std=10, is_mobile=True,
            ),
        },
        schedule_weekday=[
            (0, 8, 'home'), (8, 9, 'commute'), (9, 18, 'work'),
            (18, 19, 'commute'), (19, 24, 'home'),
        ],
        schedule_weekend=[(0, 24, 'home')],
    ),

    'suburban_student': PersonaProfile(
        name='Suburban Student (Bob)',
        locations={
            'home': LocationProfile(
                'Suburban House LTE', rssi_mean=-68, rssi_shadow_std=5,
                cell_load_base=0.30, peak_dl_mbps=50, peak_ul_mbps=20,
                latency_min_ms=22, latency_max_ms=100,
                speed_mean=0, speed_std=1,
            ),
            'campus': LocationProfile(
                'University Campus 5G', rssi_mean=-60, rssi_shadow_std=4,
                cell_load_base=0.65, peak_dl_mbps=150, peak_ul_mbps=60,
                latency_min_ms=18, latency_max_ms=110,
                speed_mean=5, speed_std=3,
            ),
            'commute': LocationProfile(
                'Highway / Car', rssi_mean=-75, rssi_shadow_std=7,
                cell_load_base=0.40, peak_dl_mbps=40, peak_ul_mbps=15,
                latency_min_ms=30, latency_max_ms=150,
                speed_mean=80, speed_std=15, is_mobile=True,
            ),
        },
        schedule_weekday=[
            (0, 9, 'home'), (9, 9.5, 'commute'),
            (9.5, 17, 'campus'), (17, 17.5, 'commute'), (17.5, 24, 'home'),
        ],
        schedule_weekend=[
            (0, 11, 'home'), (11, 11.5, 'commute'),
            (11.5, 17, 'campus'), (17, 17.5, 'commute'), (17.5, 24, 'home'),
        ],
    ),

    'rural_remote': PersonaProfile(
        name='Rural Remote Worker (Charlie)',
        locations={
            'home': LocationProfile(
                'Rural Edge LTE', rssi_mean=-88, rssi_shadow_std=6,
                cell_load_base=0.20, peak_dl_mbps=20, peak_ul_mbps=8,
                latency_min_ms=40, latency_max_ms=200,
                speed_mean=0, speed_std=1,
            ),
            'town': LocationProfile(
                'Town Centre LTE', rssi_mean=-72, rssi_shadow_std=5,
                cell_load_base=0.45, peak_dl_mbps=40, peak_ul_mbps=15,
                latency_min_ms=28, latency_max_ms=130,
                speed_mean=3, speed_std=2,
            ),
            'commute': LocationProfile(
                'Rural Road', rssi_mean=-92, rssi_shadow_std=10,
                cell_load_base=0.25, peak_dl_mbps=15, peak_ul_mbps=5,
                latency_min_ms=50, latency_max_ms=350,
                speed_mean=90, speed_std=15, is_mobile=True,
            ),
        },
        schedule_weekday=[
            (0, 9, 'home'), (9, 13, 'home'),
            (13, 14, 'commute'), (14, 16, 'town'),
            (16, 17, 'commute'), (17, 24, 'home'),
        ],
        schedule_weekend=[(0, 24, 'home')],
    ),

    'dense_urban': PersonaProfile(
        name='Dense Urban Resident (Dana)',
        locations={
            'apartment': LocationProfile(
                'City Apartment 5G', rssi_mean=-72, rssi_shadow_std=5,
                cell_load_base=0.60, peak_dl_mbps=100, peak_ul_mbps=40,
                latency_min_ms=20, latency_max_ms=90,
                speed_mean=0, speed_std=1,
            ),
            'coworking': LocationProfile(
                'Urban Co-working 5G', rssi_mean=-65, rssi_shadow_std=4,
                cell_load_base=0.50, peak_dl_mbps=150, peak_ul_mbps=60,
                latency_min_ms=15, latency_max_ms=70,
                speed_mean=0, speed_std=1,
            ),
            'commute': LocationProfile(
                'Walking / Cycling Urban', rssi_mean=-78, rssi_shadow_std=6,
                cell_load_base=0.70, peak_dl_mbps=50, peak_ul_mbps=20,
                latency_min_ms=30, latency_max_ms=150,
                speed_mean=15, speed_std=5, is_mobile=True,
            ),
        },
        schedule_weekday=[
            (0, 8, 'apartment'), (8, 8.5, 'commute'),
            (8.5, 17, 'coworking'), (17, 17.5, 'commute'),
            (17.5, 24, 'apartment'),
        ],
        schedule_weekend=[
            (0, 11, 'apartment'), (11, 13, 'commute'), (13, 24, 'apartment'),
        ],
    ),

    'frequent_traveler': PersonaProfile(
        name='Frequent Traveler (Eve)',
        locations={
            'home': LocationProfile(
                'Airport-Suburb LTE', rssi_mean=-70, rssi_shadow_std=5,
                cell_load_base=0.35, peak_dl_mbps=50, peak_ul_mbps=20,
                latency_min_ms=25, latency_max_ms=120,
                speed_mean=0, speed_std=1,
            ),
            'transit': LocationProfile(
                'Airport / Train Station', rssi_mean=-80, rssi_shadow_std=7,
                cell_load_base=0.85, peak_dl_mbps=30, peak_ul_mbps=10,
                latency_min_ms=50, latency_max_ms=300,
                speed_mean=5, speed_std=3,
            ),
            'commute': LocationProfile(
                'High-Speed Rail / Highway', rssi_mean=-85, rssi_shadow_std=10,
                cell_load_base=0.40, peak_dl_mbps=25, peak_ul_mbps=8,
                latency_min_ms=45, latency_max_ms=250,
                speed_mean=150, speed_std=30, is_mobile=True,
            ),
        },
        schedule_weekday=[
            (0, 6, 'home'), (6, 7, 'commute'), (7, 9, 'transit'),
            (9, 16, 'commute'), (16, 18, 'transit'),
            (18, 20, 'commute'), (20, 24, 'home'),
        ],
        schedule_weekend=[(0, 24, 'home')],
    ),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PersonaDataGenerator:
    """Generate per-user network traces from PERSONAS profiles."""

    # ------------------------------------------------------------------
    # Internal physics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sinr_factor(rssi_dbm: float) -> float:
        """Logistic map: RSSI → [0, 1] spectral-efficiency factor."""
        return 1.0 / (1.0 + np.exp(-(rssi_dbm - _RSSI_THRESHOLD) / _RSSI_SCALE))

    @staticmethod
    def _effective_load(base_load: float, hour: float) -> float:
        """
        Blend location baseline with empirical time-of-day load curve.
        Location contributes 60 %; time-of-day contributes 40 %.
        """
        tod = float(_LOAD_CURVE[int(hour) % 24])
        return float(np.clip(0.60 * base_load + 0.40 * tod, 0.02, 0.97))

    @staticmethod
    def _location_at(
        persona: PersonaProfile, hour: float, is_weekend: bool
    ) -> str:
        schedule = persona.schedule_weekend if is_weekend else persona.schedule_weekday
        for h_start, h_end, loc_key in schedule:
            if h_start <= hour < h_end:
                return loc_key
        return schedule[-1][2]

    def _sample_raw(
        self,
        loc: LocationProfile,
        cell_load: float,
        rng: np.random.Generator,
        in_handoff: bool = False,
    ) -> dict:
        """
        Sample one time-step of network metrics from the capacity model.

        Throughput  = peak × SINR_efficiency × load_derating × lognormal_noise
        Latency     = min + (max - min) × load²   (queuing approximation)
        Packet loss = SINR-driven loss + load-driven loss   (additive)
        Jitter      = proportional to latency × packet-loss variance
        """
        # Instantaneous RSSI with log-normal shadowing
        rssi = float(rng.normal(loc.rssi_mean, loc.rssi_shadow_std))
        if in_handoff:
            rssi -= float(rng.uniform(10, 22))   # signal drop during handoff
        rssi = float(np.clip(rssi, -115, -40))
        sinr = self._sinr_factor(rssi)

        # Throughput: non-linear load de-rating (accelerates above 70 % load)
        load_factor = max(0.04, 1.0 - 0.92 * cell_load ** 1.5)
        dl = float(np.clip(
            loc.peak_dl_mbps * sinr * load_factor * float(rng.lognormal(0, 0.08)),
            0.05, loc.peak_dl_mbps,
        ))
        ul = float(np.clip(
            loc.peak_ul_mbps * sinr * load_factor * float(rng.lognormal(0, 0.08)),
            0.02, loc.peak_ul_mbps,
        ))

        # Latency: queuing model with log-normal noise
        lat_base = loc.latency_min_ms + (loc.latency_max_ms - loc.latency_min_ms) * cell_load ** 2
        if in_handoff:
            lat_base = min(lat_base * 3.5, 600.0)
        latency = float(np.clip(
            lat_base * float(rng.lognormal(0, 0.09)),
            loc.latency_min_ms, 600.0,
        ))

        # Packet loss: SINR-driven + load-driven (both compound in poor conditions)
        pkt_loss = (1.0 - sinr) ** 2 * 14.0 + cell_load ** 3 * 10.0
        if in_handoff:
            pkt_loss = min(pkt_loss + float(rng.uniform(5, 16)), 25.0)
        pkt_loss = float(np.clip(pkt_loss + rng.normal(0, 0.25), 0.0, 25.0))

        # Jitter: scales with latency variance and packet loss
        jitter = float(np.clip(
            (1.5 + latency * 0.07 + pkt_loss * 1.4) * float(rng.lognormal(0, 0.07)),
            0.5, 120.0,
        ))

        # Mobility speed
        speed = float(np.clip(rng.normal(loc.speed_mean, loc.speed_std), 0.0, 220.0))

        # Congestion label: derived from physics, not pre-assigned.
        # Weights slightly favour cell load over SINR so that a well-connected
        # user in a congested cell correctly shows medium/high congestion.
        score = 0.45 * (1.0 - sinr) + 0.55 * cell_load
        label = 'low' if score < 0.28 else ('medium' if score < 0.52 else 'high')

        return {
            'throughput_dl':   dl,
            'throughput_ul':   ul,
            'latency':         latency,
            'packet_loss':     pkt_loss,
            'jitter':          jitter,
            'signal_strength': rssi,
            'mobility_speed':  speed,
            'congestion_true': label,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_session(
        self,
        persona_key: str,
        n_steps: int = 600,
        hour_of_day: float = 9.0,
        is_weekend: bool = False,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate one streaming session for a persona.

        The session starts at ``hour_of_day`` and advances one second per
        step.  Location transitions happen in real time if the session is
        long enough to cross a schedule boundary.  AR(1) temporal smoothing
        ensures realistic feature continuity between steps.

        Handoff events are injected randomly in mobile locations: a 3-6-
        second degraded period (signal drop, latency spike, packet loss
        burst) approximately every 90 seconds of transit time.
        """
        persona = PERSONAS[persona_key]
        rng = np.random.default_rng(seed)

        rows: list = []
        prev: Optional[dict] = None
        handoff_countdown = 0
        steps_in_mobile = 0
        _handoff_interval = 90    # expected seconds between handoffs
        _handoff_check_p  = 0.15  # probability of triggering a handoff

        for step in range(n_steps):
            hour = (hour_of_day + step / 3600.0) % 24.0
            loc_key = self._location_at(persona, hour, is_weekend)
            loc = persona.locations[loc_key]
            cell_load = self._effective_load(loc.cell_load_base, hour)

            # Handoff logic for mobile locations
            in_handoff = False
            if loc.is_mobile:
                steps_in_mobile += 1
                if handoff_countdown > 0:
                    in_handoff = True
                    handoff_countdown -= 1
                elif steps_in_mobile > _handoff_interval and rng.random() < _handoff_check_p:
                    handoff_countdown = int(rng.integers(3, 7))
                    in_handoff = True
                    steps_in_mobile = 0
            else:
                steps_in_mobile = 0
                handoff_countdown = 0

            raw = self._sample_raw(loc, cell_load, rng, in_handoff=in_handoff)

            # AR(1) smoothing on numeric features only
            if prev is None:
                smoothed = dict(raw)
            else:
                smoothed = {
                    k: (_AR1_ALPHA * prev[k] + (1.0 - _AR1_ALPHA) * raw[k]
                        if k != 'congestion_true' else raw[k])
                    for k in raw
                }

            smoothed['time']     = step
            smoothed['hour']     = round(hour, 4)
            smoothed['location'] = loc_key
            smoothed['persona']  = persona_key
            rows.append(smoothed)
            prev = {k: smoothed[k] for k in raw}

        return pd.DataFrame(rows)

    def generate_user_dataset(
        self,
        persona_key: str,
        n_sessions: int = 20,
        session_duration: int = 300,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a multi-session training dataset for one user.

        Each schedule segment (home / work / commute) gets equal representation
        regardless of its duration.  Sessions are assigned by cycling through a
        shuffled segment list, so every location appears proportionally even when
        n_sessions is small.  This ensures short but important segments (e.g. a
        one-hour commute window) are not crowded out by long stationary segments.
        """
        persona = PERSONAS[persona_key]
        rng = np.random.default_rng(seed)

        # One slot per schedule segment — equal weight, not duration-weighted
        session_pool: list = []
        for h_start, h_end, _ in persona.schedule_weekday:
            session_pool.append((h_start, h_end, False))
        for h_start, h_end, _ in persona.schedule_weekend:
            session_pool.append((h_start, h_end, True))

        # Build a shuffled index sequence that covers the pool evenly
        pool_indices: list = []
        while len(pool_indices) < n_sessions:
            pool_indices.extend(rng.permutation(len(session_pool)).tolist())
        pool_indices = pool_indices[:n_sessions]

        frames = []
        for i, pool_idx in enumerate(pool_indices):
            h_start, h_end, is_weekend = session_pool[pool_idx]
            usable_window = max(0.25, h_end - h_start - session_duration / 3600.0)
            hour = float(rng.uniform(h_start, h_start + usable_window))
            df = self.generate_session(
                persona_key,
                n_steps=session_duration,
                hour_of_day=hour,
                is_weekend=is_weekend,
                seed=int(rng.integers(0, 200_000)),
            )
            df['session'] = i
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def generate_all_personas(
        self,
        n_sessions: int = 20,
        session_duration: int = 300,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict mapping persona_key → training DataFrame."""
        rng = np.random.default_rng(seed)
        return {
            key: self.generate_user_dataset(
                key,
                n_sessions=n_sessions,
                session_duration=session_duration,
                seed=int(rng.integers(0, 200_000)),
            )
            for key in PERSONAS
        }

    def generate_streaming_trace(
        self,
        persona_key: str,
        duration: int = 300,
        hour_of_day: float = 18.0,
        is_weekend: bool = False,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Generate a single trace for ABR simulation (drop-in for NetworkSimulator)."""
        return self.generate_session(
            persona_key,
            n_steps=duration,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            seed=seed,
        )
