"""Train + calibrate XGBoost; log to MLflow."""
import argparse
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .features import build_features


def load_cfg(path):
    return yaml.safe_load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/processed/bills.parquet")
    ap.add_argument("--out", default="artifacts/model.joblib")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    df = pd.read_parquet(args.data)
    if df.empty:
        raise SystemExit("No data — run `python -m legis.data.build_dataset` first.")

    X, y = build_features(df)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=cfg["test_size"], stratify=y, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=cfg["val_size"], stratify=y_tv, random_state=42)

    mlflow.set_experiment("legis-forecast")
    with mlflow.start_run():
        mlflow.log_params(cfg["params"])
        clf = XGBClassifier(**cfg["params"])
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        cal = CalibratedClassifierCV(clf, method=cfg["calibrate"], cv="prefit")
        cal.fit(X_va, y_va)

        proba = cal.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)
        ap_score = average_precision_score(y_te, proba)
        mlflow.log_metric("test_aucpr", ap_score)
        print(classification_report(y_te, pred))
        print(f"Test AUCPR: {ap_score:.3f}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal, args.out)
        mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
