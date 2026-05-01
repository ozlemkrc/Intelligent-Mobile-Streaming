"""
Per-user PyTorch LSTM for network congestion classification.

PersonalTrainer fits a PersonalLSTM on one user's time-series data.
CrossUserEvaluator runs a controlled experiment:

  For each target user:
    - Personal model  → trained on THAT user's data
    - Generic model   → trained on ALL OTHER users' data
  Both are evaluated on the target user's held-out test set.

The resulting table shows whether personalisation beats a generic model —
which is the empirical proof the project's claim rests on.

Design choices
--------------
* Unidirectional LSTM (causal): the model only sees past observations, as
  it would in a real streaming client.
* LayerNorm on the final hidden state: stabilises training without the
  cost of batch-norm (which breaks on variable-length sequences).
* AdamW + cosine-annealing LR schedule + gradient clipping.
* Temporal train/val split — no shuffling, so the validation set is truly
  "future" data from the model's perspective.
* Early stopping on validation loss with patience.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURES = [
    'throughput_dl', 'throughput_ul', 'latency',
    'packet_loss', 'jitter', 'signal_strength', 'mobility_speed',
]
# Sorted alphabetically — this is what LabelEncoder produces
CLASSES = ['high', 'low', 'medium']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PersonalLSTM(nn.Module):
    """
    Unidirectional, 2-layer LSTM for online congestion classification.

    Input:  (batch, window, n_features)
    Output: (batch, n_classes)  — raw logits
    """

    def __init__(
        self,
        input_dim:  int   = 7,
        hidden_dim: int   = 64,
        n_layers:   int   = 2,
        n_classes:  int   = 3,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1]))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PersonalTrainer:
    """Fit and evaluate a PersonalLSTM on one user's data."""

    def __init__(
        self,
        window:       int   = 15,
        hidden_dim:   int   = 64,
        n_layers:     int   = 2,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        epochs:       int   = 30,
        batch_size:   int   = 128,
        patience:     int   = 7,
        device:       Optional[str] = None,
    ):
        self.window       = window
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.device       = self._resolve_device(device)

        self.scaler = StandardScaler()
        self.le     = LabelEncoder()
        self.le.fit(CLASSES)

        self.model:        Optional[PersonalLSTM] = None
        self.train_losses: List[float]            = []
        self.val_losses:   List[float]            = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """
        Select compute device, falling back to CPU if CUDA is unavailable
        or if the installed cuDNN kernel is incompatible with the GPU.
        """
        if device:
            return torch.device(device)
        if not torch.cuda.is_available():
            return torch.device('cpu')
        try:
            # Quick LSTM-path test — catches cuDNN kernel image mismatches.
            _t = nn.LSTM(1, 1, 1, batch_first=True).cuda()
            _t(torch.zeros(1, 1, 1).cuda())
            del _t
            return torch.device('cuda')
        except Exception:
            return torch.device('cpu')

    def _label_col(self, df: pd.DataFrame) -> str:
        return 'congestion_true' if 'congestion_true' in df.columns else 'congestion'

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding window segmentation; preserves temporal order."""
        Xs, ys = [], []
        for i in range(self.window, len(X)):
            Xs.append(X[i - self.window: i])
            ys.append(y[i])
        return (
            np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.int64),
        )

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, val_frac: float = 0.15) -> dict:
        """
        Train on user data with a temporal train / val split.

        The split is strictly temporal (no shuffling): the last ``val_frac``
        of the data acts as "future" observations the model has never seen.
        Returns a dict with training statistics.
        """
        X_raw = df[FEATURES].values.astype(np.float64)
        y_raw = self.le.transform(df[self._label_col(df)].values)

        n         = len(X_raw)
        val_start = int(n * (1.0 - val_frac))

        X_train = self.scaler.fit_transform(X_raw[:val_start])
        X_val   = self.scaler.transform(X_raw[val_start:])
        y_train, y_val = y_raw[:val_start], y_raw[val_start:]

        Xs_tr, ys_tr = self._make_sequences(X_train, y_train)
        Xs_va, ys_va = self._make_sequences(X_val,   y_val)

        t_tr = self._to_tensor(Xs_tr)
        l_tr = self._to_tensor(ys_tr)
        t_va = self._to_tensor(Xs_va)
        l_va = self._to_tensor(ys_va)

        self.model = PersonalLSTM(
            input_dim=len(FEATURES),
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_classes=len(CLASSES),
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5,
        )
        # Soft class-weighted loss: square-root of balanced weights.
        # Full 'balanced' weighting is too aggressive when a class is rare (<5%),
        # causing training instability.  Square-root weighting gently up-weights
        # minority classes (e.g. 'low' congestion in rural personas) without
        # letting rare-class loss dominate the gradient signal.
        cw = compute_class_weight('balanced', classes=np.unique(y_raw), y=y_raw)
        cw = np.sqrt(cw)
        cw = (cw / cw.mean()).astype(np.float32)
        cw_tensor = torch.FloatTensor(cw).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)

        best_val_loss  = float('inf')
        best_state:    Optional[dict] = None
        patience_left  = self.patience
        self.train_losses = []
        self.val_losses   = []

        for _ in range(self.epochs):
            self.model.train()
            perm       = torch.randperm(len(t_tr), device=self.device)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, len(t_tr), self.batch_size):
                idx = perm[start: start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(t_tr[idx])
                loss   = criterion(logits, l_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            epoch_loss /= max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(t_va), l_va).item()

            scheduler.step()
            self.train_losses.append(epoch_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        val_metrics = self._eval_tensors(t_va, ys_va)
        return {
            'epochs_trained': len(self.train_losses),
            'best_val_loss':  round(best_val_loss, 5),
            'val_accuracy':   val_metrics['accuracy'],
            'val_f1':         val_metrics['f1_macro'],
        }

    def _eval_tensors(self, X_t: torch.Tensor, y_true: np.ndarray) -> dict:
        """Evaluate the model on pre-prepared tensors."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        labels = list(range(len(CLASSES)))
        return {
            'accuracy':         float(accuracy_score(y_true, y_pred)),
            'f1_macro':         float(f1_score(y_true, y_pred, average='macro',
                                               zero_division=0, labels=labels)),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
            'y_true':           y_true,
            'y_pred':           y_pred,
            'class_names':      self.le.classes_.tolist(),
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate on a DataFrame using the already-fitted scaler.
        Call ``fit`` before calling this method.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")
        X = self.scaler.transform(df[FEATURES].values.astype(np.float64))
        y = self.le.transform(df[self._label_col(df)].values)
        Xs, ys = self._make_sequences(X, y)
        return self._eval_tensors(self._to_tensor(Xs), ys)

    def predict_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return string class labels for every row in df.
        The first ``window`` rows are predicted by padding at the beginning
        (edge-fill), matching real-world cold-start behaviour.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_series().")
        X = self.scaler.transform(df[FEATURES].values.astype(np.float64))
        self.model.eval()
        preds: List[str] = []
        with torch.no_grad():
            for i in range(len(X)):
                start = max(0, i - self.window + 1)
                seq   = X[start: i + 1]
                if len(seq) < self.window:
                    seq = np.pad(seq,
                                 ((self.window - len(seq), 0), (0, 0)),
                                 mode='edge')
                t      = torch.from_numpy(seq[np.newaxis].astype(np.float32)).to(self.device)
                idx    = int(self.model(t).argmax(dim=1).item())
                preds.append(str(self.le.inverse_transform([idx])[0]))
        return np.array(preds)

    def save(self, path: str) -> None:
        """Persist model weights and preprocessing state."""
        torch.save({
            'model_state':  self.model.state_dict() if self.model else None,
            'scaler_mean':  self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'window':       self.window,
            'hidden_dim':   self.hidden_dim,
            'n_layers':     self.n_layers,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'PersonalTrainer':
        """Restore a saved trainer from disk."""
        ckpt    = torch.load(path, map_location='cpu', weights_only=False)
        trainer = cls(
            window=ckpt['window'],
            hidden_dim=ckpt['hidden_dim'],
            n_layers=ckpt['n_layers'],
            **kwargs,
        )
        trainer.scaler.mean_            = ckpt['scaler_mean']
        trainer.scaler.scale_           = ckpt['scaler_scale']
        trainer.scaler.n_features_in_   = len(FEATURES)
        if ckpt['model_state'] is not None:
            trainer.model = PersonalLSTM(
                input_dim=len(FEATURES),
                hidden_dim=ckpt['hidden_dim'],
                n_layers=ckpt['n_layers'],
                n_classes=len(CLASSES),
            )
            trainer.model.load_state_dict(ckpt['model_state'])
        return trainer


# ---------------------------------------------------------------------------
# Cross-user experiment
# ---------------------------------------------------------------------------

class CrossUserEvaluator:
    """
    Controlled experiment: personalised model vs generic model per user.

    For each target user U:
      - Personal model  → fit on U's training split only
      - Generic model   → fit on the concatenated training splits of all
                          other users (no data from U)
    Both models are then evaluated on U's held-out test split.

    The experiment is the proof: if personalised accuracy > generic accuracy
    consistently across users, the model genuinely captures user-specific
    network behaviour rather than relying on population-level patterns.
    """

    def __init__(self, test_frac: float = 0.20, **trainer_kwargs):
        self.test_frac      = test_frac
        self.trainer_kwargs = trainer_kwargs

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cut = int(len(df) * (1.0 - self.test_frac))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    def run(
        self,
        user_datasets: Dict[str, pd.DataFrame],
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run the full cross-user evaluation.

        Parameters
        ----------
        user_datasets
            Mapping of persona_key → full DataFrame (training + test rows).
        progress_callback
            Optional callable receiving a float in [0, 1] for progress bars.

        Returns
        -------
        pd.DataFrame with columns:
            user, persona_name,
            personal_acc, generic_acc, delta_acc,
            personal_f1,  generic_f1,  delta_f1
        """
        from src.persona_generator import PERSONAS   # local import avoids circular

        train_sets: Dict[str, pd.DataFrame] = {}
        test_sets:  Dict[str, pd.DataFrame] = {}
        for uid, df in user_datasets.items():
            train_sets[uid], test_sets[uid] = self._split(df)

        user_ids   = list(user_datasets.keys())
        n_steps    = len(user_ids) * 2          # personal + generic per user
        step       = 0
        rows: list = []

        for uid in user_ids:
            # ── Personal model ───────────────────────────────────────────
            personal = PersonalTrainer(**self.trainer_kwargs)
            personal.fit(train_sets[uid])
            step += 1
            if progress_callback:
                progress_callback(step / n_steps)
            pm = personal.evaluate(test_sets[uid])

            # ── Generic model (all other users' training data) ───────────
            other_train = pd.concat(
                [train_sets[oid] for oid in user_ids if oid != uid],
                ignore_index=True,
            )
            generic = PersonalTrainer(**self.trainer_kwargs)
            generic.fit(other_train)
            step += 1
            if progress_callback:
                progress_callback(step / n_steps)
            gm = generic.evaluate(test_sets[uid])

            rows.append({
                'user':         uid,
                'persona_name': PERSONAS[uid].name,
                'personal_acc': round(pm['accuracy'], 4),
                'generic_acc':  round(gm['accuracy'], 4),
                'delta_acc':    round(pm['accuracy'] - gm['accuracy'], 4),
                'personal_f1':  round(pm['f1_macro'], 4),
                'generic_f1':   round(gm['f1_macro'], 4),
                'delta_f1':     round(pm['f1_macro'] - gm['f1_macro'], 4),
            })

        return pd.DataFrame(rows)
