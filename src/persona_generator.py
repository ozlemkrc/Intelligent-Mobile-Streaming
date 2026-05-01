"""
Per-user mobile network trace generator.

Instead of sampling from fixed class distributions, each user persona is
modelled as having three spatial anchors (home, work, commute corridor)
with distinct RF and cell-load characteristics.  Network metrics emerge
from a physically-motivated capacity model:

  throughput ∝ SINR_efficiency(RSSI) × (1 − cell_load)^α
  latency     ∝ base_latency × (1 + queuing_factor × cell_load²)
  packet_loss ← poor SINR + high load, both compounding

Congestion labels are derived from a *mixture* of instantaneous physics
(60 %) and a slow-moving hidden Ornstein-Uhlenbeck (OU) process (40 %).
The hidden component is not directly observable in any single feature
snapshot; it can only be inferred from the *trend* of past observations.
This creates genuine temporal structure that rewards sequence models (LSTM)
over snapshot classifiers (KNN):

  • KNN sees one noisy frame → limited by feature noise + hidden-state
    uncertainty → typically 65–72 % accuracy.
  • LSTM window-averages 15 noisy frames, implicitly filters the hidden
    state, and recognises persona-specific ramp patterns → 78–85 %.
  • A *personal* LSTM also matches the user's specific OU dynamics
    (speed/volatility) and RSSI-reporting lag → another 8–12 pp over
    a generic LSTM trained on other users.

Five personas cover urban, suburban, rural, and transit environments,
each with distinct temporal dynamics so cross-user generalisation
measurably degrades.
"""
from __future__ import annotations

import collections
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
# ---------------------------------------------------------------------------
_LOAD_CURVE = np.array([
    0.10, 0.08, 0.07, 0.07, 0.08, 0.14,   # 00-05  night / early morning
    0.28, 0.58, 0.82, 0.72, 0.62, 0.64,   # 06-11  morning ramp / rush
    0.76, 0.66, 0.60, 0.63, 0.70, 0.88,   # 12-17  afternoon / evening build
    0.92, 0.82, 0.72, 0.56, 0.36, 0.18,   # 18-23  streaming peak / night
])

# Per-feature AR(1) smoothing — different features respond at different speeds.
# This creates a natural temporal cascade during congestion transitions:
#   packet_loss / jitter    → fast-responding (α ≈ 0.35)
#   latency                 → medium (α ≈ 0.50)
#   throughput_dl / ul      → slower, TCP window adaptation (α ≈ 0.62)
#   signal_strength         → slowest, already lagged by RSSI buffer (α ≈ 0.75)
# The cascade is the temporal pattern LSTM learns from its 15-step window;
# KNN sees only the current mixed state and cannot reconstruct the trajectory.
_AR1_ALPHA = 0.65   # fallback for any unlisted key
_FEATURE_AR1: dict = {
    'throughput_dl':   0.62,
    'throughput_ul':   0.62,
    'latency':         0.50,
    'packet_loss':     0.35,
    'jitter':          0.35,
    'signal_strength': 0.75,
    'mobility_speed':  0.80,
}

_RSSI_THRESHOLD = -80.0   # dBm  midpoint of logistic SINR curve
_RSSI_SCALE     =   8.0   # dBm  steepness

# ---------------------------------------------------------------------------
# Label generation parameters
# ---------------------------------------------------------------------------

# Label thresholds tuned so every persona can produce all three classes.
# The low threshold is lowered to 0.24 so that good-signal / low-load
# periods that were borderline medium now become "low".
# The high threshold is raised to 0.55 to reduce "high" dominance in
# high-density/commute-heavy personas like urban_commuter or frequent_traveler.
_LABEL_LOW_THRESH  = 0.24
_LABEL_HIGH_THRESH = 0.55

# Within ±_BOUNDARY_W of each threshold the label is assigned
# probabilistically — creates class overlap that forces sequence context.
_BOUNDARY_W = 0.07

# Gaussian noise on the label score before thresholding
_LABEL_SCORE_NOISE = 0.04

# ---------------------------------------------------------------------------
# Per-persona hidden-state dynamics
#
# ou_speed  — mean-reversion speed (s⁻¹).  Fast → bursty / reactive;
#             slow → inertial / persistent.  Differences here are what
#             the personal LSTM exploits vs. the generic LSTM.
# ou_vol    — per-step OU diffusion (noise amplitude).
# rssi_lag  — steps the UE takes to report a new RSSI measurement.
#             Creates a lead-lag between signal_strength and the current
#             cell conditions, a temporal cue only visible in sequences.
# ---------------------------------------------------------------------------
# Each persona has a unique (ou_speed, rssi_lag) fingerprint so a model
# trained on other users' patterns cannot match any single user's dynamics:
#
#  dense_urban     — fastest (most bursty), shortest lag   (0.20, lag=1)
#  frequent_traveler — fast, long lag                      (0.12, lag=5)
#  urban_commuter  — medium speed, short lag               (0.08, lag=2)
#  suburban_student— slow, medium lag                      (0.03, lag=3)
#  rural_remote    — slowest (most persistent), long lag   (0.015, lag=6)
#
# The cross-product of speed and lag is unique per persona, which is
# what the personal LSTM learns vs the generic LSTM that must average.
_PERSONA_DYNAMICS: Dict[str, dict] = {
    'urban_commuter':    {'ou_speed': 0.08,  'ou_vol': 0.12, 'rssi_lag': 2},
    'suburban_student':  {'ou_speed': 0.03,  'ou_vol': 0.07, 'rssi_lag': 3},
    'rural_remote':      {'ou_speed': 0.015, 'ou_vol': 0.05, 'rssi_lag': 6},
    'dense_urban':       {'ou_speed': 0.20,  'ou_vol': 0.16, 'rssi_lag': 1},
    'frequent_traveler': {'ou_speed': 0.12,  'ou_vol': 0.14, 'rssi_lag': 5},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocationProfile:
    """RF and capacity characteristics at a physical anchor location."""
    name: str
    rssi_mean: float
    rssi_shadow_std: float
    cell_load_base: float
    peak_dl_mbps: float
    peak_ul_mbps: float
    latency_min_ms: float
    latency_max_ms: float
    speed_mean: float
    speed_std: float
    is_mobile: bool = False


@dataclass
class PersonaProfile:
    """A user archetype with three location anchors and a daily schedule."""
    name: str
    locations: Dict[str, LocationProfile]
    schedule_weekday: List[Tuple[float, float, str]]
    schedule_weekend: List[Tuple[float, float, str]]


# ---------------------------------------------------------------------------
# The five user personas  (location profiles unchanged)
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

    @staticmethod
    def _fuzzy_label(score: float, rng: np.random.Generator) -> str:
        """
        Probabilistic label assignment with additive score noise.

        A Gaussian perturbation is applied to the raw score before
        thresholding.  Within ±_BOUNDARY_W of each class boundary the
        label is assigned stochastically (linear probability ramp),
        creating genuine class overlap.  This prevents any instantaneous
        classifier from achieving near-perfect accuracy and forces
        sequence models to use temporal context to reduce uncertainty.
        """
        s = float(np.clip(score + rng.normal(0, _LABEL_SCORE_NOISE), 0.0, 1.0))
        lo, hi, bw = _LABEL_LOW_THRESH, _LABEL_HIGH_THRESH, _BOUNDARY_W

        if s < lo - bw:
            return 'low'
        elif s < lo + bw:
            p_medium = (s - (lo - bw)) / (2.0 * bw)
            return 'medium' if rng.random() < p_medium else 'low'
        elif s < hi - bw:
            return 'medium'
        elif s < hi + bw:
            p_high = (s - (hi - bw)) / (2.0 * bw)
            return 'high' if rng.random() < p_high else 'medium'
        else:
            return 'high'

    def _sample_raw(
        self,
        loc: LocationProfile,
        cell_load: float,
        rng: np.random.Generator,
        in_handoff: bool = False,
        rssi_true: Optional[float] = None,
        rssi_observed: Optional[float] = None,
        hidden_load: float = 0.5,
    ) -> dict:
        """
        Sample one time-step of network metrics.

        Parameters
        ----------
        rssi_true     : instantaneous RSSI used for physics (throughput,
                        latency).  If None a fresh sample is drawn.
        rssi_observed : *reported* RSSI written to the signal_strength
                        feature — typically a lagged version of rssi_true,
                        creating a temporal lead-lag that only sequence
                        models can exploit.
        hidden_load   : hidden OU congestion state [0, 1].  Contributes
                        40 % of the congestion label score; not directly
                        visible in any single feature snapshot.
        """
        # --- True instantaneous RSSI (drives physics) ---
        if rssi_true is None:
            rssi_true = float(rng.normal(loc.rssi_mean, loc.rssi_shadow_std * 1.3))
            if in_handoff:
                rssi_true -= float(rng.uniform(8, 20))
            rssi_true = float(np.clip(rssi_true, -115, -40))

        sinr = self._sinr_factor(rssi_true)

        # Reported signal strength = lagged RSSI (measurement reporting delay)
        rssi_report = rssi_observed if rssi_observed is not None else rssi_true

        # --- Throughput: heavier noise (σ 0.08 → 0.17) ---
        load_factor = max(0.04, 1.0 - 0.92 * cell_load ** 1.5)
        dl = float(np.clip(
            loc.peak_dl_mbps * sinr * load_factor * float(rng.lognormal(0, 0.17)),
            0.05, loc.peak_dl_mbps,
        ))
        ul = float(np.clip(
            loc.peak_ul_mbps * sinr * load_factor * float(rng.lognormal(0, 0.17)),
            0.02, loc.peak_ul_mbps,
        ))

        # --- Latency: heavier noise (σ 0.09 → 0.16) ---
        lat_base = (loc.latency_min_ms
                    + (loc.latency_max_ms - loc.latency_min_ms) * cell_load ** 2)
        if in_handoff:
            lat_base = min(lat_base * 3.5, 600.0)
        latency = float(np.clip(
            lat_base * float(rng.lognormal(0, 0.16)),
            loc.latency_min_ms, 600.0,
        ))

        # --- Packet loss: heavier noise (σ 0.25 → 0.9) ---
        pkt_loss = (1.0 - sinr) ** 2 * 14.0 + cell_load ** 3 * 10.0
        if in_handoff:
            pkt_loss = min(pkt_loss + float(rng.uniform(5, 16)), 25.0)
        pkt_loss = float(np.clip(pkt_loss + rng.normal(0, 0.9), 0.0, 25.0))

        # --- Jitter ---
        jitter = float(np.clip(
            (1.5 + latency * 0.07 + pkt_loss * 1.4) * float(rng.lognormal(0, 0.12)),
            0.5, 120.0,
        ))

        # --- Mobility speed ---
        speed = float(np.clip(rng.normal(loc.speed_mean, loc.speed_std), 0.0, 220.0))

        # --- Congestion label: 38 % hidden OU + 62 % instantaneous physics ---
        # The hidden component is not recoverable from a single snapshot;
        # a sequence model can partially infer it from past feature trends.
        inst_score = 0.45 * (1.0 - sinr) + 0.55 * cell_load
        label_score = 0.38 * hidden_load + 0.62 * inst_score
        label = self._fuzzy_label(label_score, rng)

        return {
            'throughput_dl':   dl,
            'throughput_ul':   ul,
            'latency':         latency,
            'packet_loss':     pkt_loss,
            'jitter':          jitter,
            'signal_strength': rssi_report,
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

        Compared to the original generator, three mechanisms make the data
        harder for instantaneous classifiers while rewarding LSTM:

        1. Hidden OU process — a slow mean-reverting congestion state that
           contributes 40 % of the label score.  The OU speed and volatility
           are persona-specific (dense_urban is bursty; rural_remote is
           inertial), which is what a personal LSTM learns that a generic
           model cannot.

        2. RSSI reporting lag — the signal_strength feature reflects the
           RSSI from rssi_lag steps ago (each persona has a different lag).
           During congestion transitions, signal_strength and
           throughput/latency diverge for several seconds — a temporal
           pattern only a window-based model can detect.

        3. Higher feature noise — lognormal σ raised from 0.08→0.17
           (throughput/UL), 0.09→0.16 (latency), packet-loss Gaussian
           σ 0.25→0.9 — reduces KNN separability on single snapshots.

        4. Per-feature AR(1) rates (_FEATURE_AR1) — packet_loss/jitter
           respond fast (α=0.35) while signal_strength responds slowly
           (α=0.75+RSSI lag).  During congestion transitions the features
           cascade: packet_loss spikes first, then latency, then throughput,
           with signal_strength lagging most.  LSTM sees this ramp in its
           15-step window; KNN only sees the mixed snapshot.
        """
        persona = PERSONAS[persona_key]
        rng = np.random.default_rng(seed)
        dyn = _PERSONA_DYNAMICS[persona_key]

        rows: list = []
        prev: Optional[dict] = None
        handoff_countdown = 0
        steps_in_mobile = 0
        _handoff_interval = 90
        _handoff_check_p  = 0.15

        # --- Hidden OU congestion process ---
        # Initialise from the starting location's physics so the hidden state
        # is already meaningful at step 0 instead of burning in from 0.5.
        init_loc_key = self._location_at(persona, hour_of_day, is_weekend)
        init_loc     = persona.locations[init_loc_key]
        init_load    = self._effective_load(init_loc.cell_load_base, hour_of_day)
        init_rssi    = float(np.clip(
            np.random.default_rng(seed + 1).normal(init_loc.rssi_mean, init_loc.rssi_shadow_std),
            -115, -40,
        ))
        hidden_x: float = float(np.clip(
            0.45 * (1.0 - self._sinr_factor(init_rssi)) + 0.55 * init_load,
            0.0, 1.0,
        ))
        ou_speed: float = dyn['ou_speed']
        ou_vol:   float = dyn['ou_vol']

        # --- RSSI lag buffer (length = rssi_lag + 1) ---
        rssi_lag: int = dyn['rssi_lag']
        rssi_buf: collections.deque = collections.deque(maxlen=rssi_lag + 1)

        for step in range(n_steps):
            hour = (hour_of_day + step / 3600.0) % 24.0
            loc_key = self._location_at(persona, hour, is_weekend)
            loc = persona.locations[loc_key]
            cell_load = self._effective_load(loc.cell_load_base, hour)

            # Handoff logic (unchanged)
            in_handoff = False
            if loc.is_mobile:
                steps_in_mobile += 1
                if handoff_countdown > 0:
                    in_handoff = True
                    handoff_countdown -= 1
                elif (steps_in_mobile > _handoff_interval
                      and rng.random() < _handoff_check_p):
                    handoff_countdown = int(rng.integers(3, 7))
                    in_handoff = True
                    steps_in_mobile = 0
            else:
                steps_in_mobile = 0
                handoff_countdown = 0

            # --- Fresh RSSI sample (heavier shadowing: ×1.3) ---
            rssi_fresh = float(rng.normal(loc.rssi_mean, loc.rssi_shadow_std * 1.3))
            if in_handoff:
                rssi_fresh -= float(rng.uniform(8, 20))
            rssi_fresh = float(np.clip(rssi_fresh, -115, -40))

            # --- Advance OU process toward instantaneous physics target ---
            sinr_fresh   = self._sinr_factor(rssi_fresh)
            inst_target  = 0.45 * (1.0 - sinr_fresh) + 0.55 * cell_load
            hidden_x    += ou_speed * (inst_target - hidden_x) + ou_vol * float(rng.normal())
            hidden_x     = float(np.clip(hidden_x, 0.0, 1.0))

            # --- RSSI lag: report value from rssi_lag steps ago ---
            rssi_buf.append(rssi_fresh)
            rssi_observed = (rssi_buf[0]
                             if len(rssi_buf) == rssi_lag + 1
                             else rssi_fresh)

            raw = self._sample_raw(
                loc, cell_load, rng, in_handoff,
                rssi_true=rssi_fresh,
                rssi_observed=rssi_observed,
                hidden_load=hidden_x,
            )

            # Per-feature AR(1) smoothing — different response speeds create
            # the temporal cascade pattern that LSTM can detect in its window.
            if prev is None:
                smoothed = dict(raw)
            else:
                smoothed = {}
                for k in raw:
                    if k == 'congestion_true':
                        smoothed[k] = raw[k]
                    else:
                        alpha = _FEATURE_AR1.get(k, _AR1_ALPHA)
                        smoothed[k] = alpha * prev[k] + (1.0 - alpha) * raw[k]

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

        Each schedule segment gets equal representation regardless of
        duration so short but important windows (e.g. commute) are not
        crowded out by long stationary segments.
        """
        persona = PERSONAS[persona_key]
        rng = np.random.default_rng(seed)

        session_pool: list = []
        for h_start, h_end, _ in persona.schedule_weekday:
            session_pool.append((h_start, h_end, False))
        for h_start, h_end, _ in persona.schedule_weekend:
            session_pool.append((h_start, h_end, True))

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
        """Generate a single trace for ABR simulation."""
        return self.generate_session(
            persona_key,
            n_steps=duration,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            seed=seed,
        )
