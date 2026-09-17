"""
retrieval.py

A retrieval index over historical (customer_message -> agent_reply) pairs.
This is what makes drafted replies "grounded in how the brand has
historically resolved similar issues" (deliverable #2) rather than freely
generated: for a new message, we find the most similar real customer
messages this brand has actually handled, and reuse the pattern of their
real reply.

Deliberately simple (TF-IDF + cosine similarity, scikit-learn only) so the
whole pipeline runs in minutes with no GPU, no downloaded embedding model,
and no external API -- see README "why TF-IDF and not embeddings".
"""
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "data/retrieval_index.pkl"

# Historical replies that are pure hand-off boilerplate ("we've sent you a
# DM") carry almost no reusable content for grounding a *public* draft reply.
# We keep them in the corpus (they're valid history) but down-weight them
# during retrieval so the agent doesn't just learn to parrot "check your DMs"
# for everything -- see report/REPORT.md failure analysis, mode #1.
_DM_ONLY_RE = re.compile(
    r"^(hi|hey|hello)?[^.!?]{0,20}(we'?ve|we have) (just )?(sent|replied)",
    re.I,
)


def _is_dm_only(reply: str) -> bool:
    if not isinstance(reply, str):
        return True
    return bool(_DM_ONLY_RE.match(reply.strip())) and len(reply) < 90


class RetrievalIndex:
    def __init__(self, vectorizer, matrix, df):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.df = df.reset_index(drop=True)

    @classmethod
    def build(cls, pairs_csv="data/spotify_pairs_weak.csv"):
        df = pd.read_csv(pairs_csv)
        df = df.dropna(subset=["customer_text_clean", "agent_text_clean"])
        df["is_dm_only"] = df["agent_text_clean"].map(_is_dm_only)
        vectorizer = TfidfVectorizer(
            max_features=30000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        matrix = vectorizer.fit_transform(df["customer_text_clean"])
        return cls(vectorizer, matrix, df)

    def save(self, path=INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "df": self.df}, f)

    @classmethod
    def load(cls, path=INDEX_PATH):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["vectorizer"], d["matrix"], d["df"])

    def query(self, text: str, k: int = 5, intent: str | None = None, prefer_substantive: bool = True):
        """Return the top-k most similar historical (customer, reply) pairs.

        If `intent` is given, we first try restricting candidates to rows
        whose weak_intent matches; if that yields fewer than k candidates we
        fall back to the full corpus (rare intents like content_catalog and
        feature_request_feedback are a small share of the corpus).
        """
        q = self.vectorizer.transform([text])
        pool = self.df
        pool_matrix = self.matrix
        if intent is not None:
            mask = (self.df["weak_intent"] == intent).values
            if mask.sum() >= k:
                pool = self.df[mask]
                pool_matrix = self.matrix[mask]
        sims = cosine_similarity(q, pool_matrix).ravel()
        # mild penalty for DM-only boilerplate replies so they don't crowd
        # out substantive ones when similarity scores are close
        if prefer_substantive:
            penalty = pool["is_dm_only"].values.astype(float) * 0.08
            sims = sims - penalty
        order = np.argsort(-sims)[:k]
        results = []
        for i in order:
            row = pool.iloc[i]
            results.append({
                "similarity": float(sims[i]),
                "customer_text": row["customer_text_clean"],
                "agent_reply": row["agent_text_clean"],
                "is_dm_only": bool(row["is_dm_only"]),
            })
        return results


def build_and_save():
    idx = RetrievalIndex.build()
    idx.save()
    print(f"Built retrieval index over {idx.matrix.shape[0]} pairs, "
          f"{idx.matrix.shape[1]} TF-IDF features.")


if __name__ == "__main__":
    build_and_save()
