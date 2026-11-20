# Capstone Report — Decline-risk ranking

- **Author:** Prithvi Satish
- **Lane:** freestyle: decline-risk ranking (which pages are about to lose 20%+ of their search impressions)
- **Repo:** https://github.com/Prithvi-Satish/FlyRank-Project
- **Paper:** https://prithvi-satish.github.io/FlyRank-Project/ (source: `docs/index.html`)
- **Date:** 2026-11-20

All numbers below come from `work/notebooks/capstone.ipynb`, which was re-run top to bottom for this report; `work/outputs/capstone_results.md` reproduced exactly. The deployed paper tells the same story at more length.

## 1. Problem framing

- **Decision:** which pages a content reviewer opens first, client by client. They have a few hours and hundreds of pages, so the real question is "which 10 pages first?".
- **Unit:** one page (content item).
- **Output:** a per-client ranking, each page with a decline risk, a suggested action, a confidence band and reason codes.
- **Action:** the editor reviews the top pages in order and decides whether to refresh, fix the snippet, or leave it.
- **Cost of a wrong call:**
    - a false alarm wastes editor time and may change a page that was fine;
    - a miss lets a page keep losing traffic.
- **Why ML helps:** the hand-written beliefs behind refresh rules ("old pages decay", "thin pages decay", aging page-one content) did not hold in the data (w04). The one signal that held, momentum, combines with traffic level, clicks and age in a way a fixed rule only partly captures.

## 2. Data safety

- **Data:** only the starter slice `data/raw/content_refresh_anonymized.csv` (30,000 pages × 44 columns, 32 clients). No warehouse data was used.
- **Label:** `is_declining_label` (last 30 days of impressions more than 20% below the previous 30 days); base rate 54.2%.
- **Excluded (28 of 44 columns, contract in w03):**
    - `trend_direction` / `trend_pct`: the label and its source;
    - every last-30-day column, and every 90-day total and 90-day rate (CTR, position, days with impressions, engagement, scroll, AI traffic): they overlap the label window;
    - `days_since_last_update`: 68% of pages were updated inside the label window;
    - generation metadata and derived tiers;
    - the IDs.
- **Leakage measured, not assumed:**
    - the 90-day totals nest exactly (90d = last 30 + previous 30 + days 61–90), so adding raw `impressions_90d` lifts AUC from 0.69 to 0.89;
    - the planted leak (`trend_pct`) gives 0.999;
    - shuffled labels give 0.495.
- **IDs:** `client_id` is used only for `GroupKFold`; `content_id` is never used. No product flags exist in the file.
- **Privacy:** nothing client-identifying appears in `work/` or `docs/`. All CSV outputs are gitignored, and the paper shows aggregates plus three illustrative queue rows with the client relabelled.

## 3. Baseline

The rule lives in `work/scripts/lane_baseline.py`; it was written in w04 and frozen.

| Condition | Points |
|---|---|
| Sliding into day −30 (days 31–60 more than 20% below days 61–90) | 2 |
| Visible (100+ impressions in days 31–60) | 1 |
| Under a year old | 1 |

- Pages with no prior traffic score 0, and ties are broken by traffic size.
- It's fair because:
    - it uses the same pre-day-−30 information as the model;
    - it was built from the only signal that survived the audit;
    - it was frozen before modelling;
    - it's recomputed in the same run and on the same folds as the model.
- **5-fold client-grouped results:** P@50 0.692 ± 0.131, AUC 0.670. For reference, the starter pipeline's hand-flag baseline gets P@50 0.464 under the same protocol.

## 4. Model / analysis

- **Method:** logistic regression on standardised features (`make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))`).
- **Why this model:** in w05, on a 6-client holdout, it had the best precision at the top (P@50 0.82 vs 0.74 for random forest and gradient boosting), stable folds, readable weights for reason codes, and low cost. The forest had the higher AUC (0.870) and average precision (0.739 vs 0.715), but those reward ordering the whole list.
- **Features** (`work/scripts/lane_features.py`), 16 numeric + 3 categorical:
    - search_volume, competition, cpc, has_keyword_data;
    - word_count, has_word_count;
    - age_at_prediction;
    - log impressions, clicks and sessions for days 31–60 and days 61–90;
    - impr_momentum_prior (the log ratio of the two windows);
    - ctr_prev30, prev30_empty;
    - competition_level, content_type, main_intent (one-hot encoded).
- **Left out on purpose:** everything listed in section 2.
- **Target:** a page is positive if its Google Search impressions in the last 30 days are more than 20% below the previous 30 days, predicted from information available at day −30.

## 5. Evaluation

- **Split:** 5-fold `GroupKFold` by client; every client is held out exactly once. Pages within a client share a site and tracking, so a random split rewards memorising clients (random 80/20: AUC 0.737 / P@50 0.88, vs grouped 0.695 / 0.79). There's no time split, because the data is one snapshot.

| 5-fold, grouped by client | P@20 | P@50 | P@200 | AUC |
|---|---:|---:|---:|---:|
| logistic regression | 0.820 ± 0.115 | 0.792 ± 0.125 | 0.793 ± 0.094 | 0.695 ± 0.077 |
| my rule baseline | 0.700 ± 0.127 | 0.692 ± 0.131 | 0.693 ± 0.162 | 0.670 ± 0.099 |
| reference starter baseline | 0.420 ± 0.115 | 0.464 ± 0.173 | 0.497 ± 0.169 | 0.599 ± 0.107 |
| base rate | 0.544 | 0.544 | 0.544 | — |

- **Fold by fold:** the model beats the rule at P@50 in 4 of 5 folds. It loses in fold 1, a single large client. It's ahead at every K from 10 to 300.
- **Within a client** (24 clients with 50+ pages), top-10 hit rate:

| Ranking | Declined |
|---|---|
| model | 75% |
| rule | 65% |
| client base rate | 53% |

- **Errors:**
    - about a quarter of the top-200 misses (w05 holdout) were near misses that dropped 10–20%;
    - misses cluster by client, which points to site-wide causes;
    - AUC is modest (0.60–0.77 across folds): the model orders the top well and the middle less well.
- **My own w02 target** (P@50 ≥ 0.80) is met in 3 of 5 folds, not reliably.

## 6. Interpretation

- **What the model leans on** (standardised weights and permutation importance, w05):
    - having any prior traffic (structural);
    - page age: younger pages are riskier;
    - impressions without clicks: visible but not chosen.
- **Confidence bands, checked out of fold:**

| Band | Declined | Pages |
|---|---|---|
| high | 76% | 1,220 |
| medium | 70% | 11,703 |
| low | 53% | 10,667 |
| none | 23% | 6,410 |

- **Negative results** (w04), which are results too:
    - older pages declined less (43% vs 63%);
    - thin-page effects were client mix;
    - the page-one aging flag was false (52% vs 62%).

  These beliefs shouldn't drive a refresh queue.
- **Everything is observed and directional** on one snapshot. These are patterns, not causes.

## 7. Recommendation

1. Work the queue per client, top 10 first. A global list collapses into one client.
2. Check the client before the pages. If most of its top 10 are flagged, look for a site-wide cause (migration, tracking, seasonality).
3. "refresh_review" first (sliding and visible), after checking keyword demand and who ranks now.
4. "snippet_and_intent_review" next (visible but not clicked): fix the title, description and intent before rewriting.
5. Don't queue pages for being old or short.
6. Log every refresh with its date, so a later cycle can compare refreshed and untouched pages. That's the only route to a causal claim.

**Confidence and limits:**

- it's a decision-support ranking on one snapshot of 32 clients;
- it isn't a forecast tested on a later month;
- it doesn't show that refreshing helps;
- the label is a hard −20% cut;
- results vary by client (P@50 ± 0.12).

**Monitoring:**

- PSI above 0.2 on a key feature: investigate before using the queue;
- PSI above 0.25 on the traffic features: retrain;
- the model below the rule's top-10 hit rate for 2 cycles in a row: fall back to the rule;
- the base rate moving 10+ points: re-check.

The drift check was tested by treating one half of the clients as new data; it flagged a word-count coverage difference (PSI 1.19).

## 8. Reproducibility

```bash
git clone https://github.com/Prithvi-Satish/FlyRank-Project.git
cd FlyRank-Project
pip install -r requirements.txt
# then run top to bottom, in order:
# work/notebooks/w03_data_contract.ipynb
# work/notebooks/w03_feature_leakage_check.ipynb   (writes work/scripts/lane_features.py)
# work/notebooks/w04_signal_audit.ipynb
# work/notebooks/w04_baseline_score.ipynb          (writes work/scripts/lane_baseline.py)
# work/notebooks/w05_model.ipynb
# work/notebooks/w06_validation_audit.ipynb
# work/notebooks/w07_action_playbook.ipynb
# work/notebooks/capstone.ipynb                    (runs on its own too; both .py files are committed)
```

- Seed 42 everywhere; pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1 (no deltas from `requirements.txt` needed).
- Figures go to `work/figures/` and summary tables to `work/outputs/` (the CSVs are gitignored by design).
- The paper's charts are copies in `docs/img/`.

---

> **Claims checklist:**
>
> - [x] observed / measured / directional / decision-support language throughout
> - [x] no causal claims
> - [x] nothing about predicting Google's algorithm
> - [x] no client-identifying details
> - [x] base rate shown next to every precision number
> - [x] numbers match a fresh re-run (2026-11-20)
