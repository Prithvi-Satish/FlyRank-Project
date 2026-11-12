"""Feature builder for my decline-risk lane (built in w03_feature_leakage_check, ML-05).
w04, w05, w06, w07 and the capstone import this so every notebook uses the exact same features.

Prediction moment = day -30 (start of the last 30 days). Only things known before that go in:
  days 61-90 back | days 31-60 back (prev_30d) || days 1-30 back = label window, never a feature
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# joblib prints a long "could not find physical cores" warning on windows unless this is set
# below the logical core count (physical cores are usually half the logical ones)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

SEED = 42

RAW_CANDIDATES = [
    Path("../../data/raw/content_refresh_anonymized.csv"),
    Path("data/raw/content_refresh_anonymized.csv"),
    Path("content_refresh_anonymized.csv"),
]

NUMERIC_FEATURES = [
    "search_volume", "competition", "cpc", "has_keyword_data",
    "word_count", "has_word_count",
    "age_at_prediction",
    "log_impr_prev30", "log_impr_d61_90",
    "log_clicks_prev30", "log_clicks_d61_90",
    "log_sessions_prev30", "log_sessions_d61_90",
    "impr_momentum_prior", "ctr_prev30", "prev30_empty",
]
CATEGORICAL_FEATURES = ["competition_level", "content_type", "main_intent"]


def load_raw() -> pd.DataFrame:
    path = next(p for p in RAW_CANDIDATES if p.exists())
    return pd.read_csv(path)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)

    # back out days 61-90 (the windows nest exactly, checked in the w03 data contract)
    df["impr_d61_90"] = df["impressions_90d"] - df["impressions_last_30d"] - df["impressions_prev_30d"]
    df["clicks_d61_90"] = df["clicks_90d"] - df["clicks_last_30d"] - df["clicks_prev_30d"]
    df["sessions_d61_90"] = df["sessions_90d"] - df["sessions_last_30d"] - df["sessions_prev_30d"]

    # traffic is heavy tailed, so log1p it
    df["log_impr_prev30"] = np.log1p(df["impressions_prev_30d"])
    df["log_impr_d61_90"] = np.log1p(df["impr_d61_90"])
    df["log_clicks_prev30"] = np.log1p(df["clicks_prev_30d"])
    df["log_clicks_d61_90"] = np.log1p(df["clicks_d61_90"])
    df["log_sessions_prev30"] = np.log1p(df["sessions_prev_30d"])
    df["log_sessions_d61_90"] = np.log1p(df["sessions_d61_90"])

    # momentum going INTO day -30: prev_30d vs the 30 days before it (log ratio, +1 so zeros dont blow up)
    df["impr_momentum_prior"] = np.log((df["impressions_prev_30d"] + 1) / (df["impr_d61_90"] + 1))
    df["ctr_prev30"] = np.where(
        df["impressions_prev_30d"] > 0, df["clicks_prev_30d"] / df["impressions_prev_30d"] * 100, 0.0
    )
    # nothing to drop from -> can never be labelled down (structural, known at day -30)
    df["prev30_empty"] = (df["impressions_prev_30d"] == 0).astype(int)

    # missing keyword / length data follows content_type, so flag it first, then fill
    df["has_keyword_data"] = df["search_volume"].notna().astype(int)
    df["has_word_count"] = df["word_count"].notna().astype(int)
    for col in ["search_volume", "competition", "cpc", "word_count"]:
        df[col] = df[col].fillna(0)

    df["age_at_prediction"] = df["content_age_days"] - 30
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_FEATURES].astype(float)
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], dtype=float)
    return pd.concat([numeric, dummies], axis=1)


def client_holdout(df: pd.DataFrame, test_share: float = 0.2, seed: int = SEED):
    """Same idea as scripts/03_train_model.py: shuffle the clients with seed 42, hold out 20% of them.
    Returns boolean masks (train, test) so no client is on both sides."""
    clients = df["client_id"].drop_duplicates().to_numpy()
    shuffled = np.random.default_rng(seed).permutation(clients)
    n_test = max(1, int(round(len(shuffled) * test_share)))
    test_mask = df["client_id"].isin(set(shuffled[:n_test])).to_numpy()
    return ~test_mask, test_mask


def precision_at_k(y, scores, k: int) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]
    return float(np.asarray(y)[order].mean())
