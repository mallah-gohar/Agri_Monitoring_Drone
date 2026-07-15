import os
from pathlib import Path

import joblib

import config


def model_path():
    return Path(__file__).resolve().parent.parent / "models" / "disease_clf.pkl"


def load_classifier():
    path = model_path()
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def save_classifier(clf, path=None):
    path = path or model_path()
    os.makedirs(path.parent, exist_ok=True)
    joblib.dump(clf, path)


def predict_rf(clf, features):
    if clf is None:
        return config.CELL_HEALTHY
    return int(clf.predict(features.reshape(1, -1))[0])


def predict_ndvi(ndvi, crop_type):
    thresholds = config.NDVI_THRESHOLDS.get(crop_type, config.NDVI_THRESHOLDS["Wheat"])
    if ndvi >= thresholds["healthy"]:
        return config.CELL_HEALTHY
    if ndvi >= thresholds["severe"]:
        return config.CELL_EARLY
    return config.CELL_SEVERE


def rf_confidence(clf, features):
    if clf is None or not hasattr(clf, "predict_proba"):
        return None
    probs = clf.predict_proba(features.reshape(1, -1))[0]
    return float(max(probs))
