"""FastAPI backend for Intelligent Mobile Streaming."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, uuid, threading
from typing import Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.persona_generator import PERSONAS, PersonaDataGenerator
from src.personal_model import PersonalTrainer, CrossUserEvaluator
from src.ml_classifier import FEATURES, CLASSES, NetworkClassifier
from src.streaming_engine import StreamingEngine, compute_metrics
from src.performance_evaluator import (
    compare, plot_quality_timeline, plot_buffer, plot_estimator,
    plot_throughput, plot_comparison_bars, plot_quality_distribution,
    run_multi_seed, plot_multi_seed_bars, plot_multi_seed_distribution,
    compute_win_rates,
)

app = FastAPI(title="Intelligent Mobile Streaming API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory state ────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "persona_data": None,
    "trainer": None,
    "baseline_clf": None,
    "active_persona": list(PERSONAS.keys())[0],
    "ts": None,
    "sim_rule": None, "sim_th": None, "sim_ml": None, "ml_preds": None,
    "multi_seed_df": None,
    "cross_user_results": None,
}
_jobs: dict[str, dict] = {}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _new_job() -> str:
    jid = str(uuid.uuid4())[:8]
    _jobs[jid] = {"status": "running", "progress": 0.0, "result": None, "error": None}
    return jid

def _finish_job(jid: str, result: Any):
    _jobs[jid].update(status="done", progress=1.0, result=result)

def _fail_job(jid: str, error: str):
    _jobs[jid].update(status="error", error=str(error))

def _fig_json(fig) -> dict:
    return json.loads(fig.to_json())

def _require(key: str, label: str):
    if _state[key] is None:
        raise HTTPException(400, f"{label} not available yet.")
    return _state[key]

def _to_python(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_python(v) for v in obj]
    return obj

# ── Personas & features ────────────────────────────────────────────────────────
@app.get("/api/personas")
def get_personas():
    return {k: {"name": p.name} for k, p in PERSONAS.items()}

@app.get("/api/features")
def get_features():
    return {"features": FEATURES, "classes": CLASSES}

@app.get("/api/state")
def get_app_state():
    return {
        "has_data": _state["persona_data"] is not None,
        "has_models": _state["trainer"] is not None,
        "has_simulation": _state["sim_th"] is not None,
        "has_multiseed": _state["multi_seed_df"] is not None,
        "has_crossuser": _state["cross_user_results"] is not None,
        "active_persona": _state["active_persona"],
    }

# ── Jobs ───────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]

# ── Data generation ────────────────────────────────────────────────────────────
class GenerateDataReq(BaseModel):
    n_sessions: int = 15
    session_duration: int = 300
    seed: int = 42

@app.post("/api/data/generate")
def generate_data(req: GenerateDataReq):
    jid = _new_job()
    def run():
        try:
            gen = PersonaDataGenerator()
            data = gen.generate_all_personas(
                n_sessions=req.n_sessions, session_duration=req.session_duration, seed=req.seed,
            )
            _state["persona_data"] = data
            for k in ["trainer", "baseline_clf", "ts", "sim_rule", "sim_th",
                      "sim_ml", "ml_preds", "multi_seed_df", "cross_user_results"]:
                _state[k] = None
            _finish_job(jid, {"n_total": sum(len(d) for d in data.values()), "n_users": len(PERSONAS)})
        except Exception as e:
            _fail_job(jid, str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}

@app.get("/api/data/summary")
def get_data_summary():
    pds = _require("persona_data", "Persona data")
    rows = []
    for k, df in pds.items():
        cnt = df["congestion_true"].value_counts(normalize=True)
        rows.append({
            "persona": PERSONAS[k].name, "key": k,
            "low": round(cnt.get("low", 0) * 100, 1),
            "medium": round(cnt.get("medium", 0) * 100, 1),
            "high": round(cnt.get("high", 0) * 100, 1),
            "rows": len(df),
        })
    return {"class_breakdown": rows}

@app.get("/api/data/{persona_key}/charts/distribution")
def chart_distribution(persona_key: str, feature: str = "throughput_dl"):
    import plotly.express as px
    pds = _require("persona_data", "Persona data")
    combined = pd.concat(
        [df.assign(persona_name=PERSONAS[k].name) for k, df in pds.items()], ignore_index=True
    )
    fig = px.histogram(combined, x=feature, color="persona_name", barmode="overlay",
                       opacity=0.65, nbins=60,
                       title=f"{feature.replace('_', ' ').title()} — all personas overlaid")
    fig.update_layout(height=360, legend_title="Persona")
    return _fig_json(fig)

@app.get("/api/data/{persona_key}/charts/pies")
def chart_pies(persona_key: str):
    import plotly.express as px
    pds = _require("persona_data", "Persona data")
    if persona_key not in pds:
        raise HTTPException(404, "Persona not found")
    df = pds[persona_key]
    loc = df["location"].value_counts()
    cls = df["congestion_true"].value_counts()
    fig_loc = px.pie(values=loc.values.tolist(), names=loc.index.tolist(), title="Time at each location")
    fig_loc.update_layout(height=300)
    fig_cls = px.pie(values=cls.values.tolist(), names=cls.index.tolist(),
                     color=cls.index.tolist(),
                     color_discrete_map={"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"},
                     title="Congestion class distribution")
    fig_cls.update_layout(height=300)
    return {"location": _fig_json(fig_loc), "congestion": _fig_json(fig_cls)}

@app.get("/api/data/{persona_key}/charts/scatter")
def chart_scatter(persona_key: str, features: str = "throughput_dl,latency,signal_strength,packet_loss"):
    import plotly.express as px
    pds = _require("persona_data", "Persona data")
    if persona_key not in pds:
        raise HTTPException(404, "Persona not found")
    feat_list = [f.strip() for f in features.split(",") if f.strip() in FEATURES]
    if len(feat_list) < 2:
        raise HTTPException(400, "Need at least 2 valid features")
    sample = pds[persona_key].sample(min(500, len(pds[persona_key])), random_state=1)
    fig = px.scatter_matrix(sample, dimensions=feat_list, color="congestion_true",
                            color_discrete_map={"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"})
    fig.update_traces(diagonal_visible=False, marker_size=3)
    fig.update_layout(height=560)
    return _fig_json(fig)

# ── Model training ─────────────────────────────────────────────────────────────
class TrainReq(BaseModel):
    persona_key: str
    lstm_window: int = 15
    lstm_epochs: int = 25
    lstm_patience: int = 7
    knn_k: int = 5
    rf_trees: int = 100

@app.post("/api/models/train")
def train_models(req: TrainReq):
    pds = _require("persona_data", "Persona data")
    jid = _new_job()
    def run():
        try:
            _state["active_persona"] = req.persona_key
            df = pds[req.persona_key]
            trainer = PersonalTrainer(window=req.lstm_window, epochs=req.lstm_epochs, patience=req.lstm_patience)
            _jobs[jid]["progress"] = 0.1
            trainer.fit(df)
            _state["trainer"] = trainer
            _jobs[jid]["progress"] = 0.7
            clf = NetworkClassifier(knn_k=req.knn_k, rf_trees=req.rf_trees)
            clf.train(df)
            _state["baseline_clf"] = clf
            for k in ["ts", "sim_rule", "sim_th", "sim_ml", "ml_preds", "multi_seed_df"]:
                _state[k] = None
            _finish_job(jid, {"persona": req.persona_key, "epochs": len(trainer.train_losses)})
        except Exception as e:
            _fail_job(jid, str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}

@app.get("/api/models/metrics")
def get_model_metrics():
    trainer = _require("trainer", "Trained models")
    clf = _require("baseline_clf", "Baseline classifiers")
    pds = _require("persona_data", "Persona data")
    df = pds[_state["active_persona"]]
    lstm_m = trainer.evaluate(df)
    return _to_python({
        "lstm": {
            "accuracy": lstm_m["accuracy"], "f1_macro": lstm_m["f1_macro"],
            "confusion_matrix": lstm_m["confusion_matrix"].tolist(),
            "class_names": lstm_m["class_names"],
            "train_losses": trainer.train_losses, "val_losses": trainer.val_losses,
        },
        "baselines": {
            name: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
                   "confusion_matrix": r["confusion_matrix"].tolist()}
            for name, r in clf.results.items()
        },
        "feature_importance": clf.feature_importances_,
        "n_rows": len(df), "window": trainer.window, "epochs_trained": len(trainer.train_losses),
    })

@app.get("/api/models/charts/loss")
def chart_loss():
    import plotly.express as px
    trainer = _require("trainer", "Trained models")
    curve_df = pd.DataFrame({
        "epoch": list(range(1, len(trainer.train_losses) + 1)),
        "Train loss": trainer.train_losses, "Val loss": trainer.val_losses,
    })
    fig = px.line(curve_df.melt("epoch", var_name="Split", value_name="Loss"),
                  x="epoch", y="Loss", color="Split", title="Cross-entropy loss per epoch")
    fig.update_layout(height=300)
    return _fig_json(fig)

@app.get("/api/models/charts/confusion")
def chart_confusion():
    import plotly.express as px
    trainer = _require("trainer", "Trained models")
    clf = _require("baseline_clf", "Baseline classifiers")
    pds = _require("persona_data", "Persona data")
    df = pds[_state["active_persona"]]
    lstm_m = trainer.evaluate(df)
    fig_lstm = px.imshow(lstm_m["confusion_matrix"], text_auto=True,
                         x=lstm_m["class_names"], y=lstm_m["class_names"],
                         color_continuous_scale="Blues",
                         labels={"x": "Predicted", "y": "Actual"}, title="LSTM Confusion Matrix")
    fig_lstm.update_layout(height=320, coloraxis_showscale=False)
    baseline_figs = {}
    for name, r in clf.results.items():
        fig = px.imshow(r["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"}, title=name)
        fig.update_layout(height=280, coloraxis_showscale=False)
        baseline_figs[name] = _fig_json(fig)
    fi = clf.feature_importances_
    fig_fi = px.bar(x=list(fi.values()), y=list(fi.keys()), orientation="h",
                    color=list(fi.values()), color_continuous_scale="Teal",
                    labels={"x": "Importance", "y": "Feature"})
    fig_fi.update_layout(height=300, coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    return {"lstm": _fig_json(fig_lstm), "baselines": baseline_figs, "feature_importance": _fig_json(fig_fi)}

# ── Streaming ──────────────────────────────────────────────────────────────────
class SimReq(BaseModel):
    persona_key: str
    hour: float = 8.5
    duration: int = 300
    seed: int = 7

@app.post("/api/simulation/run")
def run_simulation(req: SimReq):
    trainer = _require("trainer", "Trained models")
    gen = PersonaDataGenerator()
    ts = gen.generate_streaming_trace(req.persona_key, duration=req.duration,
                                      hour_of_day=req.hour, seed=req.seed)
    preds = trainer.predict_series(ts)
    engine = StreamingEngine()
    sim_rule = engine.simulate(ts, method="rule")
    sim_th = engine.simulate(ts, method="threshold")
    sim_ml = engine.simulate(ts, method="ml", predictions=preds)
    _state.update(ts=ts, sim_rule=sim_rule, sim_th=sim_th, sim_ml=sim_ml,
                  ml_preds=preds, multi_seed_df=None)
    return _to_python({
        "rule": compute_metrics(sim_rule),
        "threshold": compute_metrics(sim_th),
        "ml": compute_metrics(sim_ml),
    })

@app.get("/api/simulation/charts")
def get_simulation_charts():
    _require("sim_th", "Simulation")
    ts = _state["ts"]
    sr, st, sm = _state["sim_rule"], _state["sim_th"], _state["sim_ml"]
    return {
        "throughput": _fig_json(plot_throughput(ts)),
        "estimator": _fig_json(plot_estimator(st, sm, sr)),
        "quality_timeline": _fig_json(plot_quality_timeline(st, sm, sr)),
        "buffer": _fig_json(plot_buffer(st, sm, sr)),
    }

# ── Comparison ─────────────────────────────────────────────────────────────────
@app.get("/api/comparison/data")
def get_comparison_data():
    _require("sim_th", "Simulation")
    metrics = compare(_state["sim_th"], _state["sim_ml"], _state["sim_rule"])
    return {
        "metrics": _to_python(metrics),
        "bars": _fig_json(plot_comparison_bars(metrics)),
        "distribution": _fig_json(plot_quality_distribution(metrics)),
    }

# ── Multi-seed ─────────────────────────────────────────────────────────────────
class MultiSeedReq(BaseModel):
    persona_key: str
    n_runs: int = 20
    duration: int = 180

@app.post("/api/multiseed/run")
def run_multiseed(req: MultiSeedReq):
    trainer = _require("trainer", "Trained models")
    jid = _new_job()
    def run():
        try:
            df_runs = run_multi_seed(trainer, persona_key=req.persona_key,
                                     n_runs=req.n_runs, duration=req.duration,
                                     progress_callback=lambda p: _jobs[jid].update(progress=p))
            _state["multi_seed_df"] = df_runs
            _finish_job(jid, {"n_runs": req.n_runs})
        except Exception as e:
            _fail_job(jid, str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}

@app.get("/api/multiseed/data")
def get_multiseed_data(metric: str = "qoe_score", baseline: str = "Rule-Based"):
    df_runs = _require("multi_seed_df", "Multi-seed analysis")
    win_rates = compute_win_rates(df_runs, baseline=baseline)
    return {
        "bars": _fig_json(plot_multi_seed_bars(df_runs)),
        "distribution": _fig_json(plot_multi_seed_distribution(df_runs, metric)),
        "win_rates": _to_python(win_rates.to_dict(orient="records")),
    }

# ── Cross-user ─────────────────────────────────────────────────────────────────
class CrossUserReq(BaseModel):
    epochs: int = 25
    window: int = 15
    patience: int = 7

@app.post("/api/crossuser/run")
def run_crossuser(req: CrossUserReq):
    pds = _require("persona_data", "Persona data")
    jid = _new_job()
    def run():
        try:
            ev = CrossUserEvaluator(test_frac=0.20, window=req.window,
                                    epochs=req.epochs, patience=req.patience)
            res = ev.run(pds, progress_callback=lambda p: _jobs[jid].update(progress=p))
            _state["cross_user_results"] = res
            _finish_job(jid, {"n_personas": len(res)})
        except Exception as e:
            _fail_job(jid, str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}

@app.get("/api/crossuser/data")
def get_crossuser_data():
    import plotly.express as px
    res = _require("cross_user_results", "Cross-user results")
    fig = px.bar(res, x="persona_name", y=["personal_acc", "generic_acc"], barmode="group",
                 color_discrete_map={"personal_acc": "#a6e3a1", "generic_acc": "#f38ba8"},
                 labels={"value": "Accuracy", "persona_name": "Persona", "variable": "Model"},
                 title="Personal vs Generic Accuracy per User")
    fig.for_each_trace(lambda t: t.update(name="Personal" if t.name == "personal_acc" else "Generic"))
    fig.update_layout(height=380, xaxis_tickangle=-15, legend_title="Model")
    cols = ["persona_name", "personal_acc", "generic_acc", "delta_acc",
            "personal_f1", "generic_f1", "delta_f1"]
    table = res[cols].rename(columns={
        "persona_name": "Persona", "personal_acc": "Personal Acc", "generic_acc": "Generic Acc",
        "delta_acc": "Δ Acc", "personal_f1": "Personal F1", "generic_f1": "Generic F1", "delta_f1": "Δ F1",
    }).to_dict(orient="records")
    return _to_python({
        "table": table,
        "mean_delta_acc": float(res["delta_acc"].mean()),
        "personal_wins": int((res["delta_acc"] > 0).sum()),
        "total_personas": len(res),
        "chart": _fig_json(fig),
    })
