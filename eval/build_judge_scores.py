import pandas as pd
from judge_scores_manual_data import SCORES

sample = pd.read_csv("eval/results/judge_sample.csv")
rows = []
for idx, row in sample.iterrows():
    s = SCORES[idx]
    rows.append({
        "sample_idx": idx,
        "id": row["id"],
        "gold_intent": row["gold_intent"],
        "customer_text": row["customer_text"],
        "draft_reply": row["draft_reply"],
        **s,
    })
df = pd.DataFrame(rows)
df.to_csv("eval/results/judge_scores.csv", index=False)

print(f"n={len(df)}")
for dim in ["relevance", "groundedness", "tone", "safety"]:
    print(f"mean {dim}: {df[dim].mean():.2f}")
print(f"auto_sendable rate: {df['auto_sendable'].mean():.2%}")
