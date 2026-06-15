"""The ML model: a calibrated gradient-boosting classifier over technical features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .features import FEATURE_COLUMNS, build_dataset


class TradingModel:
    """Wraps a sklearn pipeline that outputs P(up) for a feature row."""

    def __init__(self, horizon: int = 5, up_threshold: float = 0.0):
        self.horizon = horizon
        self.up_threshold = up_threshold
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: List[str] = FEATURE_COLUMNS

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )

    def train(self, frames: List[pd.DataFrame], test_fraction: float = 0.2) -> dict:
        """Train on a list of per-symbol OHLCV frames. Returns a metrics dict.

        Splits chronologically per symbol (no shuffling) so we never peek at the
        future when validating.
        """
        X_train_parts, y_train_parts, X_test_parts, y_test_parts = [], [], [], []
        for df in frames:
            X, y = build_dataset(df, self.horizon, self.up_threshold)
            if len(X) < 50:
                continue
            split = int(len(X) * (1 - test_fraction))
            X_train_parts.append(X.iloc[:split])
            y_train_parts.append(y.iloc[:split])
            X_test_parts.append(X.iloc[split:])
            y_test_parts.append(y.iloc[split:])

        if not X_train_parts:
            raise RuntimeError("Not enough data to train (need >=50 usable rows per symbol).")

        X_train = pd.concat(X_train_parts)
        y_train = pd.concat(y_train_parts)
        X_test = pd.concat(X_test_parts)
        y_test = pd.concat(y_test_parts)

        self.pipeline = self._new_pipeline()
        self.pipeline.fit(X_train, y_train)

        proba = self.pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "base_rate_up": float(y_test.mean()),
        }
        # AUC is undefined if the test set is single-class.
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        return metrics

    def predict_proba_up(self, features_row: pd.DataFrame) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")
        X = features_row[self.feature_columns]
        return float(self.pipeline.predict_proba(X)[:, 1][-1])

    # --- persistence ---
    def save(self, path: Optional[Path] = None) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Nothing to save; train first.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or (MODELS_DIR / "model.joblib")
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "horizon": self.horizon,
                "up_threshold": self.up_threshold,
                "feature_columns": self.feature_columns,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TradingModel":
        path = path or (MODELS_DIR / "model.joblib")
        if not Path(path).exists():
            raise FileNotFoundError(f"No model at {path}. Run `aitrader train` first.")
        blob = joblib.load(path)
        m = cls(horizon=blob["horizon"], up_threshold=blob["up_threshold"])
        m.pipeline = blob["pipeline"]
        m.feature_columns = blob["feature_columns"]
        return m
