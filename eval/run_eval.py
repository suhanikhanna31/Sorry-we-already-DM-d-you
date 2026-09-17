"""
run_eval.py

Runs the trivial baseline, simple baseline, and main agent (local mode) over
the full golden set and computes automated metrics:
  - intent classification: accuracy, macro-F1, per-class precision/recall,
    confusion matrix
  - escalation decision: precision, recall, F1 against gold_should_escalate
    (positive class = "should escalate")
  - reply-quality proxies (automated, cheap, sanity-check only -- see
    llm_judge.py for the rubric-based quality score):
      * length in a plausible tweet-reply range (20-280 chars)
      * lexical overlap with the reference historical reply (ROUGE-1 F1),
        as a weak "did we land in the same neighborhood as a real agent"
        signal -- NOT a quality score by itself (a good reply can have zero
        word overlap with the one specific reference reply on file).

Writes:
  eval/results/predictions.csv   (one row per golden example per system)
  eval/results/metrics.json      (all the numbers used in report/REPORT.md)
"""
import json
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, "src")
from agent import SupportAgent
from baselines import TrivialBaseline, SimpleBaseline

GOLDEN = "eval/golden_set.csv"


def rouge1_f1(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    p = overlap / len(tb)
    r = overlap / len(ta)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def run_system(system, golden: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for _, g in golden.iterrows():
        out = system.handle(g["customer_text"])
        rows.append({
            "id": g["id"],
            "system": name,
            "customer_text": g["customer_text"],
            "gold_intent": g["gold_intent"],
            "pred_intent": out["intent"],
            "gold_should_escalate": g["gold_should_escalate"],
            "pred_should_escalate": out["should_escalate"],
            "escalate_reason": out["escalate_reason"],
            "draft_reply": out["draft_reply"],
            "reference_reply": g["reference_agent_reply"],
            "reply_len": len(out["draft_reply"]),
            "reply_len_ok": 20 <= len(out["draft_reply"]) <= 280,
            "rouge1_f1_vs_reference": rouge1_f1(out["draft_reply"], g["reference_agent_reply"]),
        })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, name: str) -> dict:
    m = {}
    m["intent_accuracy"] = accuracy_score(df["gold_intent"], df["pred_intent"])
    m["intent_macro_f1"] = f1_score(df["gold_intent"], df["pred_intent"], average="macro", zero_division=0)
    m["intent_classification_report"] = classification_report(
        df["gold_intent"], df["pred_intent"], zero_division=0, output_dict=True,
    )
    labels = sorted(df["gold_intent"].unique())
    cm = confusion_matrix(df["gold_intent"], df["pred_intent"], labels=labels)
    m["confusion_matrix_labels"] = labels
    m["confusion_matrix"] = cm.tolist()

    m["escalation_precision"] = precision_score(df["gold_should_escalate"], df["pred_should_escalate"], zero_division=0)
    m["escalation_recall"] = recall_score(df["gold_should_escalate"], df["pred_should_escalate"], zero_division=0)
    m["escalation_f1"] = f1_score(df["gold_should_escalate"], df["pred_should_escalate"], zero_division=0)
    m["escalation_predicted_rate"] = df["pred_should_escalate"].mean()
    m["escalation_gold_rate"] = df["gold_should_escalate"].mean()

    # The critical safety number: of the messages that SHOULD have been
    # escalated, how many did this system silently auto-handle instead?
    should = df[df["gold_should_escalate"]]
    if len(should):
        m["missed_escalations_n"] = int((~should["pred_should_escalate"]).sum())
        m["missed_escalations_rate"] = float((~should["pred_should_escalate"]).mean())
        m["missed_escalation_examples"] = (
            should[~should["pred_should_escalate"]][["id", "customer_text", "escalate_reason"]]
            .head(10).to_dict("records")
        )
    else:
        m["missed_escalations_n"] = 0
        m["missed_escalations_rate"] = 0.0

    m["reply_len_ok_rate"] = df["reply_len_ok"].mean()
    m["mean_rouge1_f1_vs_reference"] = df["rouge1_f1_vs_reference"].mean()
    return m


def main():
    golden = pd.read_csv(GOLDEN)
    print(f"Loaded {len(golden)} golden examples")

    systems = {
        "trivial": TrivialBaseline(),
        "simple": SimpleBaseline(),
        "main_agent": SupportAgent(),
    }

    all_preds = []
    metrics = {}
    for name, sys_obj in systems.items():
        print(f"Running {name}...")
        preds = run_system(sys_obj, golden, name)
        all_preds.append(preds)
        metrics[name] = compute_metrics(preds, name)
        print(f"  intent_accuracy={metrics[name]['intent_accuracy']:.3f}  "
              f"intent_macro_f1={metrics[name]['intent_macro_f1']:.3f}  "
              f"escalation_f1={metrics[name]['escalation_f1']:.3f}  "
              f"missed_escalations={metrics[name]['missed_escalations_n']}/{int(golden['gold_should_escalate'].sum())}")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv("eval/results/predictions.csv", index=False)
    with open("eval/results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\nWrote eval/results/predictions.csv and eval/results/metrics.json")


if __name__ == "__main__":
    main()
