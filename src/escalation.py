"""
escalation.py

Rule-based escalation policy: decides whether the agent may auto-post a
drafted reply, or must hand the conversation to a human agent instead.

This codifies the exact policy used to hand-label eval/golden_set.csv (see
eval/golden_labels.py docstring) so that "the agent" and "the ground truth"
are held to the same explicit standard. Every decision returns a reason
string -- deliverable #3 ("with a stated reason").

Design note: this is intentionally a transparent rule layer, not a learned
classifier. For a real deployment you'd want the false-negative rate here
(escalation-worthy messages that slip through as auto-handled) monitored and
tuned by the trust & safety / support ops team, and rules like these are far
easier for a human reviewer to audit and adjust than a black-box score. See
report/REPORT.md "what I chose not to build."
"""
import re

_SECURITY_RE = re.compile(
    r"\b(hack(ed)?|compromis\w+|unauthori[sz]ed|stolen (card|credit)|"
    r"suspicious|didn'?t (ask|request) (for )?(a )?(password reset|reset)|"
    r"someone (else )?(logged|accessed|is using))\b", re.I,
)
_BILLING_HARM_RE = re.compile(
    r"\b(charged (me )?twice|double charg|charged (again|after|despite)|"
    r"billed (again|after|for \w+ months? after)|no refund|didn'?t refund|"
    r"wrong amount|overcharg\w+|charged (the wrong|\$?\d+(\.\d+)? instead)|"
    r"debited|unauthorized (charge|payment)|"
    r"(active|working).{0,15}(account|premium|subscription).{0,15}"
    r"(cancel(led|ed)?|has been cancel))\b", re.I,
)
_PII_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}|\b(my (email|username) is)\b", re.I,
)
_PROFANITY_RE = re.compile(
    r"\b(fuck\w*|shit\w*|bullshit|asshole|damn it)\b", re.I,
)
_CHURN_RE = re.compile(
    r"\b(cancel(ing|led)? my (account|subscription|premium)( for good)?|"
    r"switch(ing)? to apple music|leaving spotify|done with spotify|"
    r"never using (this|spotify) again)\b", re.I,
)
_LONGSTANDING_RE = re.compile(
    r"\b(for (months?|years?|weeks?)|(\d+|a) (month|year|week)s? (now|ago|"
    r"in a row)|still (not|hasn'?t)|it'?s been (over )?a (year|month))\b",
    re.I,
)

LOW_CONFIDENCE_THRESHOLD = 0.35


def decide(text: str, intent: str, confidence: float) -> tuple[bool, str]:
    """Returns (should_escalate, reason)."""
    if _SECURITY_RE.search(text):
        return True, "security/account compromise signal detected"
    if _BILLING_HARM_RE.search(text):
        return True, "billing dispute - money appears to have moved incorrectly"
    if _PII_RE.search(text):
        return True, "customer exposed personal info (email/username) in a public tweet"

    profane = bool(_PROFANITY_RE.search(text))
    longstanding = bool(_LONGSTANDING_RE.search(text))
    churn = bool(_CHURN_RE.search(text))
    if profane and (longstanding or churn):
        return True, "sustained anger/profanity combined with a long-unresolved issue or churn threat"
    if churn and longstanding:
        return True, "explicit churn threat after a long-unresolved issue"

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, f"intent classifier confidence too low ({confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD}) to act with confidence"

    # A model can be *confidently* wrong about how actionable a message is:
    # very short messages classified into the catch-all bucket ("Help?",
    # "Contact Spotify") get high confidence simply because short generic
    # phrases reliably land in that bucket in training data -- that's a
    # different thing from the message containing enough information to
    # act on. Word-count is a crude but generalizable proxy for that gap
    # (see report/REPORT.md failure analysis, mode #3).
    if intent == "general_complaint_vague" and len(text.split()) <= 4:
        return True, "message too short/low-information to act on confidently despite high classifier confidence"

    return False, "routine request matching a known safe-to-automate pattern"
