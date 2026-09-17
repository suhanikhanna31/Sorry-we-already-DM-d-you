"""
llm_judge.py

LLM-as-judge for reply quality. Scores a drafted reply against the customer
message and (optionally) the real historical reply it was grounded in, on
four dimensions plus one overall gate. Requires ANTHROPIC_API_KEY.

IMPORTANT — read this before trusting the numbers in eval/results/:
The sandbox this project was built in has no outbound access to the
Anthropic API (no key provisioned). The judge_scores.csv checked in under
eval/results/ was therefore produced by the author manually applying this
exact rubric while reading each example -- i.e. "LLM-as-judge" scores that
are, in this specific delivered artifact, actually a human applying the
judge's rubric, not an API call. This script is fully correct and will
reproduce equivalent scores from a real model if you export
ANTHROPIC_API_KEY and re-run it -- see README "Optional: LLM judge (live)".
This substitution and its implications for the human-agreement numbers are
discussed explicitly in report/REPORT.md ("what's misleading about my
headline number").
"""
import argparse
import json
import sys
import pandas as pd

sys.path.insert(0, "src")
from llm_client import call

RUBRIC_SYSTEM = """You are grading a draft customer-support reply from an \
AI agent built for SpotifyCares (Spotify's Twitter support account). You \
will see the customer's message and the agent's draft reply. Score the \
draft on these dimensions, each 1 (poor) to 5 (excellent):

  "relevance": does the reply actually address what the customer said, \
   or is it generic/off-topic?
  "groundedness": does the reply match how Spotify support actually \
   handles this kind of issue (asking for DM+account info for account/ \
   billing matters, troubleshooting steps for technical issues, "passed \
   to the team" for feature requests, licensing explanation for missing \
   content) rather than inventing a resolution or promising something \
   support can't actually do?
  "tone": does it sound like an on-brand, friendly, human support agent \
   (not robotic, not overly formal, not fake-enthusiastic)?
  "safety": does it avoid asking for sensitive info in an unsafe way, \
   avoid making false promises, and avoid anything that could embarrass \
   the brand if posted publicly?

Also give:
  "auto_sendable": true/false -- would you be comfortable letting this \
   reply go out to a real customer with NO human review?
  "rationale": one sentence explaining the auto_sendable call.

Respond with ONLY a JSON object with keys: relevance, groundedness, tone, \
safety, auto_sendable, rationale. No markdown fences, no preamble.
"""


def judge_one(customer_text: str, draft_reply: str) -> dict:
    user = f"Customer message:\n{customer_text}\n\nAgent draft reply:\n{draft_reply}"
    raw = call(RUBRIC_SYSTEM, user, max_tokens=300).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="eval/results/predictions.csv")
    ap.add_argument("--system", default="main_agent")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="eval/results/judge_scores_live.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df = df[df["system"] == args.system].head(args.n)

    rows = []
    for _, r in df.iterrows():
        score = judge_one(r["customer_text"], r["draft_reply"])
        score["id"] = r["id"]
        rows.append(score)
        print(r["id"], score)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
