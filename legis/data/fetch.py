"""Download Congress.gov bulk data for one or more Congresses."""
import argparse
import os
from pathlib import Path
import requests
from tqdm import tqdm

BASE = "https://www.govinfo.gov/bulkdata/BILLSTATUS"


def fetch(congress: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for chamber in ("hr", "s"):
        url = f"{BASE}/{congress}/{chamber}/BILLSTATUS-{congress}-{chamber}.zip"
        dest = out / f"{congress}-{chamber}.zip"
        if dest.exists():
            continue
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in tqdm(r.iter_content(1 << 14), desc=dest.name):
                    f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    for c in args.congress:
        fetch(c, args.out)


if __name__ == "__main__":
    main()
