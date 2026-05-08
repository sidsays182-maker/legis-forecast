"""Score new bills with a trained model."""
import joblib
import pandas as pd
from .features import build_features


def predict(df: pd.DataFrame, model_path: str = "artifacts/model.joblib"):
    model = joblib.load(model_path)
    X, _ = build_features(df)
    return model.predict_proba(X)[:, 1]
