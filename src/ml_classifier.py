from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = [
    'throughput_dl', 'throughput_ul', 'latency',
    'packet_loss', 'jitter', 'signal_strength', 'mobility_speed'
]
CLASSES = ['low', 'medium', 'high']


class NetworkClassifier:
    """Wraps KNN and Random Forest classifiers for network congestion prediction."""

    def __init__(self, knn_k: int = 5, rf_trees: int = 100, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.le.fit(CLASSES)

        self.models: dict = {
            'KNN': KNeighborsClassifier(n_neighbors=knn_k),
            'Random Forest': RandomForestClassifier(
                n_estimators=rf_trees,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        self.results: dict = {}
        self._trained = False

    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        X = df[FEATURES].values
        y = self.le.transform(df['congestion'].values)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        self.results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred,
                target_names=self.le.classes_,
                output_dict=True,
                zero_division=0,
            )
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'report': report,
                'y_test': y_test,
                'y_pred': y_pred,
            }
        self._trained = True
        # Feature importances (RF only)
        rf = self.models['Random Forest']
        self.feature_importances_ = dict(zip(FEATURES, rf.feature_importances_))
        return self.results

    def predict(self, features: np.ndarray, model_name: str = 'Random Forest') -> np.ndarray:
        """Predict congestion class labels for raw (unscaled) feature rows."""
        if not self._trained:
            raise RuntimeError("Call train() before predict().")
        X = self.scaler.transform(np.atleast_2d(features))
        encoded = self.models[model_name].predict(X)
        return self.le.inverse_transform(encoded)

    def predict_series(self, df: pd.DataFrame, model_name: str = 'Random Forest') -> np.ndarray:
        """Predict for every row in a time-series DataFrame."""
        return self.predict(df[FEATURES].values, model_name)

    def evaluate_on_trace(
        self, ts: pd.DataFrame, model_name: str = 'Random Forest'
    ) -> dict:
        """Classify the streaming trace and score against congestion_true.

        Distinct from the i.i.d. test-set evaluation in train(): the trace is
        sequential and drawn from a different sampling process (time-correlated,
        scenario-cycled), so accuracy here is the defensible 'does the ML→ABR
        story hold in practice?' number.
        """
        preds = self.predict_series(ts, model_name=model_name)
        true  = ts['congestion_true'].values
        return {
            'accuracy':         accuracy_score(true, preds),
            'f1_macro':         f1_score(true, preds, average='macro',
                                         zero_division=0, labels=CLASSES),
            'confusion_matrix': confusion_matrix(true, preds, labels=CLASSES),
            'y_true':           true,
            'y_pred':           preds,
        }


# ------------------------------------------------------------------
# Optional LSTM classifier
# ------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def lstm_available() -> bool:
    return _TF_AVAILABLE


class LSTMClassifier:
    """Sliding-window LSTM for sequential congestion classification."""

    def __init__(self, window: int = 10, epochs: int = 20, batch_size: int = 32):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMClassifier.")
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.le.fit(CLASSES)
        self._model = None
        self.history = None

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.window, len(X)):
            Xs.append(X[i - self.window:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        # Accept time-series frames (label column 'congestion_true') or
        # i.i.d. dataset frames (label column 'congestion'). For an LSTM
        # the former is what's actually meaningful — the i.i.d. frame is
        # shuffled so its temporal dimension is noise.
        label_col = 'congestion_true' if 'congestion_true' in df.columns else 'congestion'
        X_raw = df[FEATURES].values
        y = self.le.transform(df[label_col].values)

        split = int(len(X_raw) * (1 - test_size))
        X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
        y_train, y_test = y[:split], y[split:]

        # Fit the scaler on TRAIN ONLY — fitting on the full array leaks
        # test-set statistics into the model.
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        Xs_train, ys_train = self._build_sequences(X_train, y_train)
        Xs_test, ys_test = self._build_sequences(X_test, y_test)

        n_classes = len(CLASSES)
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(self.window, len(FEATURES)), return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = model.fit(
            Xs_train, ys_train,
            validation_data=(Xs_test, ys_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        self._model = model

        y_pred_prob = model.predict(Xs_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = accuracy_score(ys_test, y_pred)
        f1 = f1_score(ys_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(ys_test, y_pred)
        return {'accuracy': acc, 'f1_macro': f1, 'confusion_matrix': cm,
                'y_test': ys_test, 'y_pred': y_pred}

    def predict_series(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[FEATURES].values)
        labels = []
        # For the first `window` steps, fall back to last prediction
        for i in range(len(X)):
            start = max(0, i - self.window + 1)
            seq = X[start:i + 1]
            if len(seq) < self.window:
                seq = np.pad(seq, ((self.window - len(seq), 0), (0, 0)), mode='edge')
            prob = self._model.predict(seq[np.newaxis], verbose=0)[0]
            labels.append(self.le.inverse_transform([np.argmax(prob)])[0])
        return np.array(labels)
