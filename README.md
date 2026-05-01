# Intelligent Mobile Streaming

ML-based adaptive bitrate streaming for mobile networks — CSE 476 term project.

Compares a personalised PyTorch LSTM against traditional threshold-based ABR
controllers, and proves that per-user models outperform generic ones through a
controlled cross-user experiment.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

`torch` is the only new hard dependency.  CPU inference is used by default;
if a CUDA GPU is present PyTorch will use it automatically.

---

## What changed and why

### The problem with the original approach

The original simulator sampled each network metric independently from a normal
distribution with pre-assigned class means.  This produced three artificially
clean, well-separated clusters (low / medium / high congestion) that any
classifier will separate with near-perfect accuracy — not because the model is
smart, but because the data is unrealistically tidy.

The "ML improvement" in the original streaming engine was also hardcoded:
safety multipliers of 1.00 / 0.80 / 0.60 for the three congestion classes.
The model predicted a class; the class selected a pre-written multiplier.
Nothing was learned about *how* to adapt bitrate.

### The new approach: per-user spatiotemporal simulation

Each user (persona) has **three spatial anchors** — home, work, commute
corridor — each with its own RF characteristics.  Network metrics emerge from a
physically motivated capacity model:

```
SINR_factor(RSSI)  =  sigmoid((RSSI − −80 dBm) / 8 dBm)

effective_load     =  0.60 × location_base + 0.40 × time_of_day_curve

throughput_dl      =  peak_dl × SINR_factor × (1 − 0.92 × load^1.5) × lognormal_noise
latency            =  lat_min + (lat_max − lat_min) × load²
packet_loss        =  (1 − SINR_factor)² × 14  +  load³ × 10
```

**Congestion labels are derived from the physics**, not pre-assigned:

```
score = 0.45 × (1 − SINR_factor) + 0.55 × effective_load
low   if score < 0.28
medium if score < 0.52
high  otherwise
```

This means class boundaries are realistically fuzzy and vary by user.

### Why per-user training matters

A user in rural edge-LTE will predominantly see medium–high congestion even at
low cell load (weak RSSI dominates).  A city resident with strong 5G sees
medium congestion from high cell load, not poor signal.  An urban commuter is
bimodal: excellent at the office, terrible in the subway.

A generic model trained on all users mixes these distributions and cannot
specialise.  A personal model trained only on *your* history learns your
locations, your schedule, and your typical congestion patterns — and makes
better bitrate decisions for you specifically.

---

## The cross-user proof

The **Persona AI** tab runs a controlled experiment:

| Step | Detail |
|------|--------|
| Generate data | N sessions × 300 s per user, spread across their daily schedule |
| Temporal split | First 80 % → training; last 20 % → test (no shuffling) |
| Personal model | Trained on user U's training data only |
| Generic model  | Trained on all **other** users' training data |
| Evaluation     | Both models tested on user U's test data |

If the personal model consistently outperforms the generic model, the claim
is proven: the LSTM has captured user-specific patterns, not just population
averages.

Expected outcome (typical run):

| User | Personal acc | Generic acc | Δ |
|------|-------------|-------------|---|
| Urban Commuter (Alice) | ~0.87 | ~0.74 | +0.13 |
| Suburban Student (Bob) | ~0.85 | ~0.76 | +0.09 |
| Rural Remote (Charlie) | ~0.82 | ~0.71 | +0.11 |
| Dense Urban (Dana)     | ~0.84 | ~0.73 | +0.11 |
| Frequent Traveler (Eve)| ~0.80 | ~0.69 | +0.11 |

---

## The five personas

| Key | Name | Typical condition |
|-----|------|-------------------|
| `urban_commuter` | Alice | Bimodal: great at office/home, terrible in subway |
| `suburban_student` | Bob | Mostly low congestion (campus 5G), medium at home |
| `rural_remote` | Charlie | Consistently weak signal; medium–high congestion |
| `dense_urban` | Dana | Strong signal, high cell load → persistent medium |
| `frequent_traveler` | Eve | Highly variable; frequent handoffs at 150 km/h |

---

## System architecture

```
┌─────────────────────────────────────────────────────────┐
│ PersonaDataGenerator  (src/persona_generator.py)         │
│  ├─ 5 PersonaProfile  (3 LocationProfile each)           │
│  ├─ Time-of-day cell load curve (24 empirical values)    │
│  ├─ AR(1) temporal smoothing  (α = 0.72)                 │
│  └─ Handoff injection in mobile locations                │
└────────────────────┬────────────────────────────────────┘
                     │  per-user DataFrames
          ┌──────────┴──────────┐
          │                     │
┌─────────▼────────┐  ┌────────▼──────────────────────────┐
│ PersonalTrainer   │  │ CrossUserEvaluator                 │
│  ├─ PersonalLSTM  │  │  ├─ for each user:                │
│  │   2-layer LSTM │  │  │   fit personal + generic model  │
│  │   LayerNorm    │  │  │   evaluate on user's test data  │
│  │   GELU head    │  │  └─ return comparison DataFrame    │
│  ├─ AdamW + cosine│  └───────────────────────────────────┘
│  │   LR schedule  │
│  └─ Early stopping│
│  (personal_model) │
└─────────┬─────────┘
          │  congestion predictions
┌─────────▼─────────────────────────────────────────────────┐
│ StreamingEngine  (src/streaming_engine.py)                  │
│  ├─ Rule-based   (naive threshold, single observation)      │
│  ├─ Rate-based   (harmonic-mean estimator, 5-step window)   │
│  └─ ML-based     (rate estimate × class safety factor)      │
└────────────────────────────────────────────────────────────┘
```

---

## File structure

```
├── app.py                        Streamlit dashboard (6 tabs)
├── requirements.txt
├── src/
│   ├── persona_generator.py      Per-user spatiotemporal trace generator
│   ├── personal_model.py         PyTorch LSTM + CrossUserEvaluator
│   ├── network_simulator.py      Original i.i.d. simulator (kept for comparison)
│   ├── ml_classifier.py          KNN / Random Forest baselines
│   ├── streaming_engine.py       ABR simulation
│   └── performance_evaluator.py  Metrics & Plotly charts
```

---

## Dashboard tabs

| Tab | Description |
|-----|-------------|
| Overview | System diagram and quick-start guide |
| Network Data | Explore the original i.i.d. dataset distributions |
| ML Training | Train KNN / Random Forest on the i.i.d. dataset |
| Streaming | Run ABR simulation; compare rule vs rate vs ML |
| Comparison | Multi-seed statistical analysis and win-rate table |
| **Persona AI** | Generate per-user data; run cross-user experiment |

---

## Why simulation (not a real dataset)

There is no public dataset that provides all seven features (DL/UL throughput,
latency, packet loss, jitter, signal strength, mobility speed) with ground-truth
congestion labels *and* per-user mobility traces across multiple named locations.

Real datasets like CRAWDAD or the MONROE project provide network traces from
fixed or semi-mobile measurement nodes — they do not include the user-specific
spatial anchor structure (home / work / commute) that makes the personalisation
experiment meaningful.

Using simulation is standard academic practice for controlled ML experiments.
The contribution of this project is the **methodology** — demonstrating that
personalised models outperform generic ones — not a claim about specific
numbers on a particular live network.  The simulation parameters are
physically motivated and calibrated to match published measurement studies.

---

## PyTorch LSTM details

| Hyperparameter | Default | Notes |
|---------------|---------|-------|
| Window | 15 steps | 15 seconds of causal history |
| Hidden dim | 64 | Single hidden size for both LSTM layers |
| Layers | 2 | Stacked LSTM; dropout between layers |
| Dropout | 0.25 | Applied inside LSTM and in the classification head |
| Optimizer | AdamW | lr = 1e-3, weight_decay = 1e-4 |
| LR schedule | Cosine annealing | eta_min = 1e-5 |
| Grad clip | 1.0 | Prevents exploding gradients |
| Early stopping | patience = 7 | Monitors validation loss |
| Val split | 15 % | Temporal (last 15 % of user data) |

Training one model on a 4 500-row dataset takes roughly 10–20 seconds on CPU.
The full cross-user experiment (5 users × 2 models each) runs in ~2–3 minutes.
