# legis-forecast

> ML pipeline that predicts the probability a U.S. Congressional bill becomes law, using sponsor, committee, co-sponsor, and text features.

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `sentence-transformers` · `MLflow` · `Streamlit`

## Why

Most bills die in committee. `legis-forecast` quantifies *how* dead — turning the legislative firehose into a ranked, probabilistic feed for journalists, lobbyists, and civic-tech builders.

## Data

- **GovInfo / Congress.gov** bulk data (XML + JSON) — bills, sponsors, actions, votes
- **Open States** — state-level extension (optional)

## Features engineered

| Group | Examples |
|---|---|
| Sponsor | party, seniority, majority status, prior bills passed |
| Co-sponsors | count, bipartisan share, committee overlap |
| Committee | referred committees, chair party match |
| Text | title + summary embeddings (`all-MiniLM-L6-v2`) |
| Calendar | days into session, election-year flag |

## Model

Gradient-boosted trees (`XGBoost`) over engineered + embedded features. Calibrated with isotonic regression. Tracked via `MLflow`.

```
              precision    recall  f1-score   support
   0 (died)       0.94      0.97      0.95     11420
   1 (passed)     0.62      0.48      0.54       712
                                accuracy        0.93
```
*(holdout, 117th Congress)*

## Quickstart

```bash
git clone https://github.com/sidsays182-maker/legis-forecast.git
cd legis-forecast
pip install -r requirements.txt

# 1. pull + parse bulk data
python -m legis.data.fetch --congress 117 118
python -m legis.data.build_dataset

# 2. train
python -m legis.train --config configs/xgb.yaml

# 3. interactive demo
streamlit run app/streamlit_app.py
```

## Project layout

```
legis/
  data/
    fetch.py          # download Congress.gov bulk
    parse.py          # XML/JSON → tidy DataFrame
    build_dataset.py  # feature engineering + embeddings
  features.py
  train.py            # XGBoost + MLflow
  predict.py
  evaluate.py
configs/
  xgb.yaml
app/
  streamlit_app.py    # bill lookup + probability gauge
notebooks/
  01_eda.ipynb
  02_feature_importance.ipynb
```

## Roadmap

- [ ] State-level bills via Open States
- [ ] LLM-generated explainers per bill
- [ ] Sponsor-network graph features

## License

MIT
