"""
build_golden_candidates.py

Draws a stratified candidate pool for the golden evaluation set. We stratify
on the *weak* intent label so rare intents (content_catalog,
feature_request_feedback) aren't drowned out by the huge default bucket, then
a human reviewer (see eval/labeling_notes.md) reads every single candidate
and assigns the final gold intent + escalation label from scratch -- the weak
label is only used to make sure the sample isn't 90% "vague complaint".

We also stratify a slice by month so the golden set isn't dominated by a
single burst of near-duplicate tweets (e.g. an outage day).
"""
import pandas as pd

SRC = "data/spotify_pairs_weak.csv"
OUT = "eval/golden_candidates.csv"
PER_INTENT = 35  # 6 intents * 35 = 210 candidates -> reviewer keeps ~180-200

df = pd.read_csv(SRC)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
df["month"] = df["created_at"].dt.to_period("M").astype(str)

parts = []
rng = 13
for intent, grp in df.groupby("weak_intent"):
    n = min(PER_INTENT, len(grp))
    # sample spread across months within the intent to avoid burst duplicates
    sampled = (
        grp.groupby("month", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n // grp["month"].nunique())), random_state=rng))
    )
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=rng)
    elif len(sampled) < n:
        extra = grp.drop(sampled.index).sample(min(n - len(sampled), len(grp) - len(sampled)), random_state=rng)
        sampled = pd.concat([sampled, extra])
    parts.append(sampled)

pool = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
pool = pool[[
    "customer_tweet_id", "customer_text_clean", "agent_text_clean",
    "weak_intent", "created_at",
]]
pool.to_csv(OUT, index=False)
print(f"Wrote {len(pool)} candidates to {OUT}")
print(pool["weak_intent"].value_counts())
