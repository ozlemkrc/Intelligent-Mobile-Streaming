# Intelligent Mobile Streaming

ML-based adaptive bitrate streaming vs threshold-based methods in mobile networks.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Optionally uncomment `tensorflow` in `requirements.txt` to enable the LSTM classifier.

## Usage

1. **Generate Dataset** — set samples per class and seed in the sidebar
2. **Train Models** — configure KNN/RF hyperparameters and train
3. **Run Simulation** — choose duration and ML model, then simulate a streaming session
4. Explore results across the **Streaming** and **Comparison** tabs

## Structure

```
├── app.py                   # Streamlit dashboard
├── src/
│   ├── network_simulator.py # Synthetic network trace generation
│   ├── ml_classifier.py     # KNN, Random Forest, optional LSTM
│   ├── streaming_engine.py  # ABR simulation (threshold vs ML)
│   └── performance_evaluator.py  # Metrics & Plotly charts
└── requirements.txt
```

## Modules

| Module | Description |
|--------|-------------|
| Network Simulator | Generates throughput, latency, packet loss, jitter, signal strength, mobility speed for low/medium/high congestion |
| ML Classifier | KNN and Random Forest trained to predict congestion class; optional LSTM for time-series |
| Streaming Engine | Buffer-based ABR simulation comparing threshold rules against ML predictions |
| Performance Evaluator | Rebuffering events, quality switches, adaptation accuracy, quality distribution |
