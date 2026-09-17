"""
build_silver_corpus.py

Applies the weak-label heuristic (intents.weak_label) to the cleaned
(customer_message -> agent_reply) pairs produced by prepare_data.py. This
"silver" (weak-labeled, noisy) corpus is what src/classifier.py trains on,
what src/retrieval.py builds its grounding index over, and what
eval/build_golden_candidates.py stratifies to draw the golden-set candidate
pool. The weak label is never treated as ground truth -- see
eval/golden_set.csv / eval/golden_labels.py for the actual hand-labeled
evaluation data.

Input:  data/spotify_pairs.csv       (written by src/prepare_data.py)
Output: data/spotify_pairs_weak.csv  (same rows, + a `weak_intent` column)
"""
import pandas as pd
from intents import weak_label

IN_PATH = "data/spotify_pairs.csv"
OUT_PATH = "data/spotify_pairs_weak.csv"


def main():
    df = pd.read_csv(IN_PATH)
    df["weak_intent"] = df["customer_text_clean"].map(weak_label)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print(df["weak_intent"].value_counts())


if __name__ == "__main__":
    main()
