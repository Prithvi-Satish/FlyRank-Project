"""My rule baseline (built in w04_baseline_score, ML-07). Frozen once written: w05 onwards import it,
so the baseline is recomputed in the same run as the model and never quietly re-tuned.

The rule in plain words (straight out of the w04 signal audit, momentum was the only signal that held in both client halves):
"A page goes to the top of the review list if it was visible before (100+ impressions in days 31-60), it was already
sliding going into day -30 (days 31-60 more than 20% below days 61-90), and its still under a year old. Pages that tick
all of that are ranked by how much traffic is at stake, biggest first. Pages with zero impressions in days 31-60 go to
the bottom, they cant decline."

Points: sliding = 2, visible = 1, under a year old = 1, so 0-4 points. No fitted weights, nothing learned from labels.
Ties inside a points level are broken by impressions_prev_30d (bigger page first).

Reason codes a page can get (more than one at once):
  already_sliding   days 31-60 impressions more than 20% below days 61-90
  visible_before    100+ impressions in days 31-60
  still_settling    page is under 365 days old (older pages declined less in the audit)
  recent_spike      days 31-60 more than double days 61-90 (could fall back)
  low_volume        1-99 impressions in days 31-60, small numbers swing a lot
  no_prior_traffic  zero impressions in days 31-60, cant be labelled down

Actions: refresh_review (sliding + visible), light_check (sliding but small), watch_spike (recent spike, check
seasonality first), no_action (no prior traffic), monitor (everything else).
"""
import numpy as np
import pandas as pd

RULE_INPUTS = ["impressions_prev_30d", "impr_d61_90", "content_age_days"]


def baseline_points(frame: pd.DataFrame) -> pd.Series:
    sliding = frame["impressions_prev_30d"] < 0.8 * frame["impr_d61_90"]
    visible = frame["impressions_prev_30d"] >= 100
    young = frame["content_age_days"] < 365
    can_decline = frame["impressions_prev_30d"] > 0
    return (can_decline * (2 * sliding + visible + young)).astype(int)


def baseline_score(frame: pd.DataFrame) -> pd.Series:
    # tie-break inside a points level: bigger page first (a percentile, so it never jumps a whole point)
    return baseline_points(frame) + 0.99 * frame["impressions_prev_30d"].rank(pct=True)


def reason_codes(frame: pd.DataFrame) -> pd.Series:
    prev, before = frame["impressions_prev_30d"], frame["impr_d61_90"]
    rules = {
        "already_sliding": prev < 0.8 * before,
        "visible_before": prev >= 100,
        "still_settling": frame["content_age_days"] < 365,
        "recent_spike": prev > 2 * before.clip(lower=1),
        "low_volume": (prev > 0) & (prev < 100),
        "no_prior_traffic": prev == 0,
    }
    codes = pd.Series("", index=frame.index)
    for name, hit in rules.items():
        codes = codes.where(~hit, codes + "|" + name)
    return codes.str.strip("|").replace("", "general_review")


def suggested_action(frame: pd.DataFrame) -> np.ndarray:
    codes = reason_codes(frame)
    return np.select(
        [codes.str.contains("no_prior_traffic"),
         codes.str.contains("already_sliding") & codes.str.contains("visible_before"),
         codes.str.contains("already_sliding"),
         codes.str.contains("recent_spike")],
        ["no_action", "refresh_review", "light_check", "watch_spike"],
        "monitor",
    )
