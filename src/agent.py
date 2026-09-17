"""
agent.py

The SpotifyCares support agent. Ties together:
  1. classify(text)   -> intent + confidence           (src/classifier.py)
  2. retrieve(text)    -> top-k similar historical pairs (src/retrieval.py)
  3. draft(intent, r)  -> grounded reply draft           (src/draft.py)
  4. decide(text, ...) -> auto-handle vs escalate + reason (src/escalation.py)

Default mode is fully local/offline (no API key needed) -- this is what the
README's "reproduce in under 15 minutes" instructions run. An optional
--mode llm path additionally asks Claude to classify + draft using the same
retrieved examples as few-shot grounding, for anyone with an API key who
wants to compare against the free/offline path.
"""
import argparse
import json
import sys

from classifier import IntentClassifier
from retrieval import RetrievalIndex
from draft import draft_reply
import escalation


class SupportAgent:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.index = RetrievalIndex.load()

    def handle(self, text: str, k: int = 5) -> dict:
        intent, confidence = self.classifier.predict(text)
        retrieval_results = self.index.query(text, k=k, intent=intent)
        draft = draft_reply(intent, retrieval_results)
        should_escalate, reason = escalation.decide(text, intent, confidence)
        return {
            "customer_text": text,
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "draft_reply": draft["draft"],
            "grounded_in": draft["grounded_in"],
            "grounding_similarity": round(draft["similarity"], 3),
            "should_escalate": should_escalate,
            "escalate_reason": reason,
        }


def main():
    ap = argparse.ArgumentParser(description="Run the SpotifyCares support agent on a message.")
    ap.add_argument("text", nargs="?", help="Customer message. If omitted, reads from stdin.")
    ap.add_argument("--mode", choices=["local", "llm"], default="local")
    args = ap.parse_args()

    text = args.text or sys.stdin.read().strip()

    if args.mode == "llm":
        from llm_agent import handle_llm
        result = handle_llm(text)
    else:
        agent = SupportAgent()
        result = agent.handle(text)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
