import pandas as pd
from golden_labels import LABELS

pool = pd.read_csv("eval/golden_candidates.csv")
assert len(pool) == len(LABELS), (len(pool), len(LABELS))

rows = []
for idx, row in pool.iterrows():
    intent, escalate, reason = LABELS[idx]
    rows.append({
        "id": idx,
        "customer_tweet_id": row["customer_tweet_id"],
        "customer_text": row["customer_text_clean"],
        "reference_agent_reply": row["agent_text_clean"],
        "weak_intent_seed": row["weak_intent"],
        "gold_intent": intent,
        "gold_should_escalate": escalate,
        "gold_escalate_reason": reason,
    })

out = pd.DataFrame(rows)
out.to_csv("eval/golden_set.csv", index=False)
print(f"Wrote eval/golden_set.csv with {len(out)} rows")
print(out["gold_intent"].value_counts())
print("escalate rate:", out["gold_should_escalate"].mean())
