"""
draft.py

Turns a retrieved historical (customer_message -> agent_reply) pair into a
draft reply for a *new* customer message. This is intentionally an
adaptation of real historical text, not free generation -- it's the
"grounded in how the brand has historically resolved similar issues" part
of the assignment (deliverable #2).

What "adaptation" means here, concretely:
  1. Strip the retrieved reply's agent-initials signature (e.g. " /JI") --
     that's an internal routing tag, not part of the message.
  2. Strip any name-personalized greeting from the retrieved reply (e.g.
     "Hey Emily!") since we don't know the new customer's name -- replace
     with a small set of generic openers that vary by intent so replies
     don't all sound identical.
  3. Leave the substantive middle of the reply untouched -- that's the
     actual "how we solve this" content being reused.

This is a template/retrieval system, not an LLM -- so it will sometimes
produce a reply that's a slightly awkward fit (see report/REPORT.md failure
analysis, mode #2, "grounded-but-generic reuse"). An LLM-mode alternative
that uses the same retrieved examples as few-shot grounding for a generated
reply is available in src/llm_client.py + src/agent.py (--mode llm) for
anyone running this with an API key.
"""
import re

_SIGNATURE_RE = re.compile(r"\s*/[A-Z]{1,4}\s*$")
_GREETING_RE = re.compile(
    r"^(hey|hi|hello)\s+[A-Z][a-zA-Z]{1,15}[!,.]?\s*", re.I,
)
_GENERIC_OPENERS = {
    "account_billing": "Hi there! Thanks for flagging this.",
    "technical_playback": "Hey! Sorry about that — happy to help.",
    "content_catalog": "Hey there!",
    "feature_request_feedback": "Hi! Thanks for the suggestion.",
    "general_complaint_vague": "Hi there, thanks for reaching out.",
    "praise_smalltalk_closed": "Hey!",
}


def _clean_reply(reply: str, intent: str) -> str:
    r = _SIGNATURE_RE.sub("", reply).strip()
    if _GREETING_RE.match(r):
        r = _GREETING_RE.sub("", r).strip()
        opener = _GENERIC_OPENERS.get(intent, "Hi there!")
        r = f"{opener} {r}"
    return r


def draft_reply(intent: str, retrieval_results: list[dict]) -> dict:
    """Pick the best retrieved candidate and adapt it into a draft.

    Returns a dict with the draft text and the provenance (which historical
    example it was grounded in + similarity score) so a human reviewer can
    audit *why* the bot said what it said -- this is what makes the reply
    "grounded" rather than a black box.
    """
    if not retrieval_results:
        return {
            "draft": "Hi there! Thanks for reaching out — could you tell us a bit more about what's going on so we can help?",
            "grounded_in": None,
            "similarity": 0.0,
        }
    # Prefer the highest-similarity candidate that isn't pure DM-boilerplate;
    # fall back to the top candidate if all top-k are boilerplate.
    substantive = [r for r in retrieval_results if not r["is_dm_only"]]
    best = substantive[0] if substantive else retrieval_results[0]
    draft = _clean_reply(best["agent_reply"], intent)
    return {
        "draft": draft,
        "grounded_in": best["customer_text"],
        "similarity": best["similarity"],
    }
