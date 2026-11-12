import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import CURRENT_YEAR, build_features  # noqa: F401  (needed so joblib can unpickle)

st.set_page_config(page_title="Citation Predictor", page_icon="📄", layout="centered")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


B = load()

st.title("📄 Research Paper Citation Predictor")
st.caption("Estimates how many citations a paper is likely to receive, based on its metadata and text.")

with st.form("paper"):
    title = st.text_input("Title *", placeholder="e.g. Deep learning for protein structure prediction")
    abstract = st.text_area("Abstract", height=160, help="Optional, but improves the prediction.")
    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Publication year", 2015, CURRENT_YEAR, 2020)
    refs = c2.number_input("Number of references", 0, 5000, int(B["ref_median"]))
    authors = c3.number_input("Number of authors", 1, 100, 4)

    journal_choice = st.selectbox("Journal", ["(other / not listed)"] + B["journals"])
    journal_other = ""
    if journal_choice == "(other / not listed)":
        journal_other = st.text_input("Journal name (optional)")
    concepts_sel = st.multiselect("Concepts / topics", B["concepts"],
                                  help="Pick the topics that best describe the paper.")
    go = st.form_submit_button("Predict citations", type="primary", use_container_width=True)

if go:
    if not title.strip():
        st.error("Please enter a title.")
        st.stop()
    journal = journal_other.strip() if journal_choice.startswith("(") else journal_choice
    row = pd.DataFrame([dict(
        title=title, abstract=abstract, publication_year=year,
        reference_count=min(refs, B["ref_cap"]), author_count=authors,
        journal=journal or "Unknown", concepts=", ".join(concepts_sel))])
    log_pred = float(B["model"].predict(build_features(row, B["journal_freq"]))[0])
    est = int(round(np.expm1(log_pred)))
    lo = int(round(np.expm1(log_pred + B["q10"])))
    hi = int(round(np.expm1(log_pred + B["q90"])))
    age = max(CURRENT_YEAR - year, 1)

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated citations", f"{est:,}")
    m2.metric("Likely range (80%)", f"{lo:,} – {hi:,}")
    m3.metric("Per year", f"{est / age:,.0f}")

    if journal and journal not in B["journal_freq"].index:
        st.info("This journal wasn't in the training data, so it was treated as unknown.")
    if not abstract.strip():
        st.info("No abstract given. Adding one usually gives a better estimate.")

    st.warning(
        "**Read this as a relative estimate.** The model was trained only on *highly cited* papers "
        "(1,136+ citations, 2015–2025), so it answers: *if this paper becomes highly cited, "
        f"roughly how much?* Its holdout R² is only {B['r2']:.2f}, so treat the number as a rough "
        "guide, not a forecast.")
