"""
llm_agent.py

Optional LLM-mode path: uses the same retrieval index as the local agent to
pull grounding examples, then asks Claude to classify + draft + decide
escalation in one structured call. Requires ANTHROPIC_API_KEY.

We still use the local retrieval index for grounding (rather than trusting
the LLM to know "how SpotifyCares has handled this before") so the
LLM-mode reply is grounded in the same real historical data as the local
mode -- the comparison between modes is then about generation quality, not
about which one has access to real examples.
"""
import json
from intents import INTENTS
from retrieval import RetrievalIndex
from llm_client import call

SYSTEM = f"""You are a support-ticket triage assistant for SpotifyCares, \
Spotify's Twitter support account. You will be given a new customer message \
plus a few similar historical (customer message -> real agent reply) pairs \
for grounding. Respond with ONLY a JSON object (no markdown fences, no \
preamble) with these keys:
  "intent": one of {INTENTS}
  "draft_reply": a short (<280 char), on-brand public reply, written in \
SpotifyCares' voice (friendly, casual, uses "Hey!"/"Hi there!", never says \
"as an AI"). If the historical examples show the real team asking for \
account details via DM for this kind of issue, do the same.
  "should_escalate": true/false -- true only if this needs a human because \
of account security compromise, money that already moved incorrectly, \
exposed personal info in the tweet, sustained anger/profanity combined with \
a long-unresolved issue, or a genuinely ambiguous message you can't act on \
confidently.
  "escalate_reason": short string explaining the escalation decision (empty \
string if should_escalate is false).
"""


def handle_llm(text: str, k: int = 4) -> dict:
    index = RetrievalIndex.load()
    examples = index.query(text, k=k)
    example_block = "\n\n".join(
        f"Customer: {e['customer_text']}\nAgent: {e['agent_reply']}" for e in examples
    )
    user = f"Historical examples:\n{example_block}\n\nNew customer message:\n{text}"
    raw = call(SYSTEM, user, max_tokens=400)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    result["customer_text"] = text
    result["grounded_in"] = [e["customer_text"] for e in examples]
    return result
