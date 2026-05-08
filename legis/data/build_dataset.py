"""Parse raw bill XML into a tidy DataFrame with engineered features."""
from pathlib import Path
import pandas as pd

# NOTE: a real implementation parses BILLSTATUS XML; for portfolio purposes we
# load a pre-computed parquet so the rest of the pipeline is reproducible.

RAW = Path("data/raw")
OUT = Path("data/processed/bills.parquet")


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        return pd.read_parquet(OUT)

    # placeholder — replace with real parser
    df = pd.DataFrame({
        "bill_id": [], "congress": [], "chamber": [],
        "title": [], "summary": [],
        "sponsor_party": [], "sponsor_seniority": [], "sponsor_majority": [],
        "n_cosponsors": [], "bipartisan_share": [],
        "n_committees": [], "chair_party_match": [],
        "days_into_session": [], "election_year": [],
        "passed": [],
    })
    df.to_parquet(OUT)
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape)
