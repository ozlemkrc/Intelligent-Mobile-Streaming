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
    """KNN and Random Forest baselines trained on persona data."""

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

    def _label_col(self, df: pd.DataFrame) -> str:
        return 'congestion_true' if 'congestion_true' in df.columns else 'congestion'

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        X = df[FEATURES].values
        y = self.le.transform(df[self._label_col(df)].values)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
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
                'accuracy':         accuracy_score(y_test, y_pred),
                'f1_macro':         f1_score(y_test, y_pred, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'report':           report,
                'y_test':           y_test,
                'y_pred':           y_pred,
            }
        self._trained = True
        rf = self.models['Random Forest']
        self.feature_importances_ = dict(zip(FEATURES, rf.feature_importances_))
        return self.results

    def predict(self, features: np.ndarray, model_name: str = 'Random Forest') -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Call train() before predict().")
        X = self.scaler.transform(np.atleast_2d(features))
        return self.le.inverse_transform(self.models[model_name].predict(X))

    def predict_series(self, df: pd.DataFrame, model_name: str = 'Random Forest') -> np.ndarray:
        return self.predict(df[FEATURES].values, model_name)
