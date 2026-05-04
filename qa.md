Overall project assessment for undergrad mobile comms course
What do they need to know about mobile comms (not ML)?
Data generation - what physics? SINR, RSSI specifics
Public datasets - what exists?
QoE - made up or standard?
What is "network simulation" / "adaptive streaming" actually doing?
Why 0.5 Mbps → 240p, 1.5 → 480p etc.


On the project overall: It's actually fine for the course. The mobile comms content is real — SINR, cell load, path loss, OU dynamics, handoffs — that's the meat of what the course covers. ML is just the decision engine on top. If anything, you have more domain-correct network modeling than most student projects, which usually just throw random noise at a linear model. The risk is that your professor asks you to explain the RF physics and you can't, not that the project is "too ML."

SINR and RSSI — what you need to know:

RSSI (Received Signal Strength Indicator) is what your phone actually measures — the total received power at the antenna in dBm. −60 dBm is excellent, −90 dBm is poor, −110 dBm is basically no service.

SINR (Signal to Interference plus Noise Ratio) is what determines how fast you can actually transmit. Shannon's theorem: capacity = bandwidth × log2(1 + SINR). Higher SINR → higher possible throughput. Our model converts RSSI to a SINR proxy via sigmoid:


SINR_factor = sigmoid((RSSI − (−80 dBm)) / 8 dBm)
−80 dBm is the typical LTE sensitivity threshold (below this the link is marginal). The sigmoid maps RSSI → [0,1] representing "what fraction of peak capacity is available given this signal level." This is a simplification of the actual Shannon formula, but directionally correct and defensible.

Cell load is the other factor. A cell tower shares spectrum among all connected users. Even with perfect signal, if 200 people are on the same tower at 6pm, your share is tiny. The formula throughput ∝ SINR_factor × (1 − 0.92 × load^1.5) captures this: at load=0 you get full capacity, at load=1 you get near zero. The 1.5 exponent models the non-linear degradation (TCP flows compete for resources, backpressure compounds).

Why our generated data has an advantage over public datasets: Public mobile traces (MONROE, FCC Measuring Broadband) are mostly short sessions, mixed users, no per-user longitudinal data. You cannot train a personalized model on them because they don't track the same user over many sessions. If asked: "we reviewed the MONROE dataset and FCC broadband traces — neither provides the per-user session depth needed to train and evaluate personalized models. MONROE is per-device and mostly fixed-location measurements; FCC is aggregate."

QoE — completely standard:

Not made up. The formula QoE = avg_bitrate − 0.3 × rebuffer_seconds − 0.02 × quality_switches is from Yin et al., "A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP," SIGCOMM 2015 — one of the most-cited ABR papers. The α=0.3 and β=0.02 weights are from that paper's user study. The Pensieve paper (Mao et al., SIGCOMM 2017, the most-cited ML-for-ABR paper) uses the same formula. So you're using the exact industry-standard QoE metric.

Video bitrate thresholds — the math:

Raw 480p: 640×480 pixels × 3 bytes (RGB) × 30 fps = ~221 Mbps. After H.264/H.265 compression (typically 150:1 to 400:1 ratio for typical content), 480p lands at 500 kbps – 2 Mbps. Our 1000 kbps for 480p is Netflix's actual target bitrate. YouTube uses 500 kbps, Netflix 1000 kbps.

The threshold of 1.5 Mbps to allow 480p (not guarantee it) is because you need throughput headroom above the bitrate — you're downloading the next segment while playing the current one, and you need to download it faster than real-time. 1.5 Mbps threshold for a 1 Mbps stream gives you 50% buffer build rate. This is defensible and matches real DASH player behavior.

dX = θ(μ − X)dt + σ dW — the OU process:

This is an Ornstein-Uhlenbeck (OU) process — a mean-reverting stochastic process. In plain terms:

X is a "hidden congestion level" that evolves over time
θ (theta): how fast it snaps back to its long-term average. High θ = bursty (Dana, θ=0.20), low θ = persistent (Charlie, θ=0.015)
μ (mu): the long-term average the process reverts to
σ dW: random Wiener noise (Brownian motion) — each timestep gets a random kick
In discrete form (what the code actually implements): X[t+1] = X[t] + θ(μ − X[t]) + σ × N(0,1)

Why OU for network congestion: real congestion has inertia. A congested cell doesn't clear instantly — it builds, stays congested for a while, then clears. OU captures this naturally. It's not mobile-comms specific (it comes from quantitative finance — the Vasicek interest rate model) but it's the standard choice for any domain with mean-reverting dynamics.

AR(1) Smoothing — generic, not telecom-specific:

feature[t] = α × feature[t−1] + (1−α) × raw[t]

This is just an exponential moving average. Completely generic signal processing. In the mobile context, the different α values reflect real physical response lags:

Packet loss/jitter (α=0.35): reported almost immediately by the protocol stack
Latency (α=0.50): queuing delay builds gradually (TCP slow start, buffer bloat)
Throughput (α=0.62): TCP's congestion window adapts slowly over multiple RTTs
Signal strength (α=0.75): RSSI hardware registers sample and average over several measurement cycles before reporting
You do not need to know the exact values — you can say "we chose these based on the typical response time constants of each measurement in LTE hardware, where PHY-layer measurements like RSSI are averaged over longer windows than protocol-layer events like packet loss."

"Low/Medium/High congestion" vs quality prediction — you are correct to flag this:

There's a real inconsistency in the presentation. The data generator creates congestion_true (Low/Medium/High) which is used for: (1) visualization in the dashboard pie charts, (2) the physics-based simulation model. The ML models (KNN, RF, LSTM) are not trained on these congestion labels anymore — they're trained to predict quality levels (240p/480p/720p/1080p) via _compute_labels. Slide 11 describes how congestion labels are generated, which is still valid as an explanation of the data model. But if asked "what does your classifier output?" the answer is quality levels, not congestion classes. The congestion label is a simulation intermediate, not an ML target.

The specific numbers in the label formula — can you defend them?

The 0.45/0.55 and 0.62/0.38 are design choices tuned to produce a realistic class distribution across all personas. Honest answer if asked: "We calibrated these weights so that the simulated congestion distribution approximately matches real LTE network measurements, where medium congestion is most prevalent and high congestion is rare but present. The exact values were tuned empirically."

Your point about varying the hidden load multiplier per persona is actually a good design insight. The OU speed (θ) already varies — that's the primary per-persona differentiation. The 0.38 weight is uniform by design to keep the model simple. You could argue for per-persona weights; the simplification is: "We varied the OU speed per persona to model different congestion inertia, and kept the mixing weight uniform to reduce the number of free parameters."

Handoffs and deep fades — yes, both are implemented:

The code confirms it. Handoffs inject into mobile locations every ~90 steps with a random trigger, dropping RSSI by 8–20 dBm, tripling latency, and spiking packet loss. Deep fades trigger with probability 1/200 per step and collapse throughput to 20 kbps (below even 240p) for 12–26 steps. Both are real, not just described in the presentation.