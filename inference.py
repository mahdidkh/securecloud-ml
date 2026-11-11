"""Inference interface for the SecureCloud AI backend (no training code, no labels needed).

    from inference import AnomalyDetector
    det = AnomalyDetector("model.joblib")
    det.predict([{"correlation_id": "w-1", "event_count": 14, "guardduty_count": 6, ...}])

Input  : one dict (or a list of dicts) with the 7 window features listed in `det.features`
         (+ optional `correlation_id`, echoed back).
Output : list of {correlation_id, anomaly_score, is_anomaly, priority, top_features}.
"""
import joblib
import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(self, path="model.joblib"):
        b = joblib.load(path)
        self.model, self.features = b["model"], list(b["features"])
        self.threshold, self.scale = float(b["threshold"]), float(b["scale"])
        self.ref_mean = np.asarray(b["ref_mean"], float); self.ref_std = np.asarray(b["ref_std"], float)

    def predict(self, events):
        if isinstance(events, dict):
            events = [events]
        df = pd.DataFrame(events)
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = df[self.features].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite numbers (no NaN/inf).")
        scores = -self.model.score_samples(X)
        z = (X - self.ref_mean) / self.ref_std
        out = []
        for i, s in enumerate(scores):
            anomaly = bool(s >= self.threshold)
            priority = "high" if s >= self.threshold + 2 * self.scale else ("medium" if anomaly else "none")
            top = np.argsort(-np.abs(z[i]))[:2]
            out.append({"correlation_id": df["correlation_id"].iloc[i] if "correlation_id" in df else i,
                        "anomaly_score": round(float(s), 5), "is_anomaly": anomaly, "priority": priority,
                        "top_features": [{"feature": self.features[j], "z_vs_normal": round(float(z[i, j]), 2)} for j in top]})
        return out
