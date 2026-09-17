"""
classifier.py

Trains a TF-IDF + multinomial Logistic Regression intent classifier on the
*silver* (weak-labeled) corpus. This is deliberately not trained on the
golden set -- the golden set is held out purely for evaluation, so the
accuracy numbers in eval/results are a genuine test of generalization from
noisy weak labels to hand-verified ground truth, not memorization.

This same trained classifier is used both as the "simple baseline" (intent
classification + one fixed canned template per intent, see
eval/baselines.py) and as the intent-routing component inside the main
agent (src/agent.py) -- the two systems differ in what they do *after*
classifying the intent (templated vs. retrieval-grounded reply, and a
richer escalation policy), not in how they classify.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "data/intent_classifier.pkl"


def train(pairs_csv="data/spotify_pairs_weak.csv"):
    df = pd.read_csv(pairs_csv).dropna(subset=["customer_text_clean", "weak_intent"])
    X_train, X_val, y_train, y_val = train_test_split(
        df["customer_text_clean"], df["weak_intent"], test_size=0.05,
        random_state=13, stratify=df["weak_intent"],
    )
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    clf.fit(Xtr, y_train)

    print("Held-out silver-label validation report (sanity check only -- "
          "real evaluation is against the hand-labeled golden set):")
    print(classification_report(y_val, clf.predict(Xva)))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)
    print(f"Saved classifier to {MODEL_PATH}")


class IntentClassifier:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            d = pickle.load(f)
        self.vectorizer = d["vectorizer"]
        self.clf = d["clf"]

    def predict(self, text: str):
        x = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(x)[0]
        classes = self.clf.classes_
        order = np.argsort(-proba)
        top = classes[order[0]]
        confidence = float(proba[order[0]])
        return top, confidence


if __name__ == "__main__":
    train()
