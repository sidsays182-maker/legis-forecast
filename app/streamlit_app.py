"""Streamlit demo: paste a bill, get a probability + drivers."""
import pandas as pd
import streamlit as st

from legis.predict import predict

st.set_page_config(page_title="legis-forecast", layout="centered")
st.title("legis-forecast")
st.caption("Probability a U.S. bill becomes law")

with st.form("bill"):
    title = st.text_input("Bill title")
    summary = st.text_area("Summary", height=160)
    chamber = st.selectbox("Chamber", ["hr", "s"])
    party = st.selectbox("Sponsor party", ["D", "R", "I"])
    seniority = st.slider("Sponsor seniority (terms)", 1, 20, 3)
    cosp = st.slider("Co-sponsors", 0, 200, 8)
    bipart = st.slider("Bipartisan share", 0.0, 1.0, 0.2)
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{
        "title": title, "summary": summary,
        "chamber": chamber, "sponsor_party": party,
        "sponsor_seniority": seniority, "sponsor_majority": True,
        "n_cosponsors": cosp, "bipartisan_share": bipart,
        "n_committees": 1, "chair_party_match": True,
        "days_into_session": 100, "election_year": False,
    }])
    p = float(predict(df)[0])
    st.metric("Probability of passage", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
