"""Shared feature code, used by BOTH train_model.py and app.py so they never drift apart."""
import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
NUM = ["publication_year", "log_refs", "log_authors", "abstract_len", "title_len", "concept_count",
       "has_abstract", "title_has_colon", "title_has_question", "is_review", "is_tool", "journal_freq"]
FEATS = NUM + ["journal", "concepts", "text"]


def split_concepts(s):
    return [c.strip() for c in str(s).split(",") if c.strip()]


def build_features(df, journal_freq):
    """df needs: title, abstract, publication_year, reference_count, author_count, journal, concepts."""
    d = df.copy()
    d["title"] = d["title"].fillna("").astype(str)
    d["abstract"] = d["abstract"].fillna("").astype(str)
    d["concepts"] = d["concepts"].fillna("").astype(str)
    d["journal"] = d["journal"].fillna("Unknown").astype(str)
    d["has_abstract"] = (d.abstract.str.len() > 0).astype(int)
    d["abstract_len"] = d.abstract.str.split().str.len()
    d["title_len"] = d.title.str.split().str.len()
    d["concept_count"] = d.concepts.apply(lambda s: len(split_concepts(s)))
    d["log_refs"] = np.log1p(d.reference_count)
    d["log_authors"] = np.log1p(d.author_count)
    d["title_has_colon"] = d.title.str.contains(":").astype(int)
    d["title_has_question"] = d.title.str.contains(r"\?").astype(int)
    d["is_review"] = d.title.str.contains(
        r"review|meta-analysis|guideline|statistics|survey|consensus", case=False).astype(int)
    d["is_tool"] = d.title.str.contains(
        r"software|toolkit|package|database|algorithm|tool\b|server|platform", case=False).astype(int)
    d["text"] = d.title + " " + d.abstract
    d["journal_freq"] = d["journal"].map(journal_freq).fillna(0)
    return d[FEATS]
