"""Run once:  python train_model.py   ->  creates model.joblib"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from features import NUM, build_features, split_concepts
import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")
for c in ["title", "abstract", "concepts", "journal"]:
    df[c] = df[c].fillna("")
y = np.log1p(df["citation_count"])


def make_model():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("jour", TargetEncoder(target_type="continuous", random_state=42), ["journal"]),
        ("conc", CountVectorizer(tokenizer=split_concepts, lowercase=False,
                                 token_pattern=None, min_df=20, binary=True), "concepts"),
        ("text", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                                      stop_words="english", min_df=3, sublinear_tf=True)),
            ("svd", TruncatedSVD(50, random_state=42))]), "text"),
    ], sparse_threshold=0)
    return Pipeline([("p", prep), ("m", RandomForestRegressor(
        300, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=42))])


# 1) holdout run -> honest R2 + residual spread for prediction intervals
tr, te, ytr, yte = train_test_split(df, y, test_size=0.2, random_state=42)
jf = tr["journal"].value_counts()
m = make_model().fit(build_features(tr, jf), ytr)
pred = m.predict(build_features(te, jf))
resid = yte.values - pred
r2 = r2_score(yte, pred)
q10, q90 = np.quantile(resid, [0.10, 0.90])
print(f"Holdout R2 = {r2:.3f} | residual 10%/90% = {q10:.3f} / {q90:.3f}")

# 2) refit on ALL data for the deployed model
jf_all = df["journal"].value_counts()
final = make_model().fit(build_features(df, jf_all), y)

top_journals = jf_all[jf_all >= 3].index.tolist()
top_concepts = (df["concepts"].str.split(", ").explode().value_counts().head(200).index.tolist())
joblib.dump(dict(model=final, journal_freq=jf_all, journals=sorted(top_journals),
                 concepts=top_concepts, r2=r2, q10=q10, q90=q90,
                 ref_cap=float(df["reference_count"].max()),
                 ref_median=float(df["reference_count"].median())), "model.joblib")
print("saved model.joblib")
