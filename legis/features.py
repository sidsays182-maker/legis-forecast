"""Feature builder: numeric + categorical + text embeddings."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NUMERIC = [
    "sponsor_seniority", "n_cosponsors", "bipartisan_share",
    "n_committees", "days_into_session",
]
CATEGORICAL = ["chamber", "sponsor_party", "sponsor_majority",
               "chair_party_match", "election_year"]
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_features(df: pd.DataFrame, embedder: SentenceTransformer | None = None):
    embedder = embedder or SentenceTransformer(EMBEDDER_NAME)
    text = (df["title"].fillna("") + ". " + df["summary"].fillna("")).tolist()
    emb = embedder.encode(text, batch_size=64, show_progress_bar=False) if text else np.empty((0, 384))
    num = df[NUMERIC].astype(float).fillna(0).values
    cat = pd.get_dummies(df[CATEGORICAL].astype(str), dummy_na=True).values
    X = np.hstack([num, cat, emb]) if len(df) else np.empty((0, num.shape[1] + cat.shape[1] + 384))
    y = df["passed"].astype(int).values if "passed" in df else None
    return X, y
