"""
prepare_data.py

Reads the raw Kaggle "Customer Support on Twitter" dump (twcs.csv) and builds
a clean, English-only set of (customer_message -> brand_reply) pairs for a
single brand: SpotifyCares.

Why SpotifyCares (see report/REPORT.md "Problem framing" for full reasoning):
  - Large enough volume (~43k tweets) to build both a retrieval corpus and a
    held-out golden set.
  - Overwhelmingly English, consumer-facing, and the public replies are
    substantive (not just "please call this number") often enough to ground
    real reply drafting.
  - Support issues cluster into a small, stable set of intents: account
    access, billing/subscription, playback/technical, content availability,
    feature requests/feedback, and praise -- a good testbed for an intent
    taxonomy without needing 77 fine-grained Banking77-style buckets.

Output: data/spotify_pairs.csv with columns:
  customer_tweet_id, customer_text, customer_text_clean, agent_tweet_id,
  agent_text, agent_text_clean, created_at
"""
import re
import sys
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic langdetect

BRAND = "SpotifyCares"
RAW_PATH = "data_raw/archive-2/twcs/twcs.csv"
OUT_PATH = "data/spotify_pairs.csv"

MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = MENTION_RE.sub("", t)
    t = URL_RE.sub("", t)
    t = t.replace("&amp;", "&")
    t = WS_RE.sub(" ", t).strip()
    return t


def is_english(t: str) -> bool:
    t = t.strip()
    if len(t) < 3:
        return False
    try:
        return detect(t) == "en"
    except Exception:
        return False


def main():
    print(f"Loading {RAW_PATH} (this is a ~500MB file, may take ~30-60s)...")
    df = pd.read_csv(RAW_PATH, dtype=str)
    df["inbound"] = df["inbound"] == "True"
    df["tweet_id"] = df["tweet_id"].astype(str)

    # Index all tweets by id for fast lookup of the customer message a brand
    # reply was responding to.
    by_id = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["author_id"] == BRAND) & (~df["inbound"])].copy()
    print(f"Raw {BRAND} tweets: {len(brand_replies)}")

    rows = []
    missing = 0
    for _, r in brand_replies.iterrows():
        parent_id = r["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id == "":
            missing += 1
            continue
        if parent_id not in by_id.index:
            missing += 1
            continue
        parent = by_id.loc[parent_id]
        # in_response_to can sometimes match multiple rows if ids collide;
        # guard against that by taking the first row.
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if not parent["inbound"]:
            # Brand replying to itself (thread continuation) -- skip, we only
            # want the first-line customer->agent exchange for grounding.
            continue
        rows.append(
            {
                "customer_tweet_id": parent["tweet_id"],
                "customer_text": parent["text"],
                "agent_tweet_id": r["tweet_id"],
                "agent_text": r["text"],
                "created_at": r["created_at"],
            }
        )

    print(f"Paired: {len(rows)}  (unpaired/missing parent: {missing})")
    out = pd.DataFrame(rows)
    out["customer_text_clean"] = out["customer_text"].map(clean_text)
    out["agent_text_clean"] = out["agent_text"].map(clean_text)

    # Drop empties / near-empties after cleaning.
    out = out[
        (out["customer_text_clean"].str.len() >= 5)
        & (out["agent_text_clean"].str.len() >= 5)
    ]

    # Language filter on the customer side (langdetect is slow -- ~40k rows
    # takes a couple minutes, which is why prepare_data is cached to disk and
    # only needs to run once).
    print("Language-filtering customer messages (English only)...")
    is_en = out["customer_text_clean"].map(is_english)
    out = out[is_en]
    print(f"After English filter: {len(out)}")

    # De-duplicate identical customer messages (bots / repeated test tweets)
    out = out.drop_duplicates(subset=["customer_text_clean"])
    print(f"After dedup: {len(out)}")

    out = out.sort_values("created_at")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
