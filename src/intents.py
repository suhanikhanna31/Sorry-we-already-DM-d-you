"""
intents.py

Defines the intent taxonomy for the SpotifyCares agent, inductively derived
from reading ~300 sampled customer messages (see report/REPORT.md). Six
intents were chosen because they each map to a *distinct handling strategy*
by the real support team, not just a distinct topic:

  account_billing        -> always verified privately (DM) before any account
                             or payment action is taken; never resolved in
                             public reply.
  technical_playback      -> triaged with public troubleshooting steps first
                             (restart, log out/in, check device/OS/version);
                             escalated to DM only if that fails or device
                             details are needed.
  content_catalog         -> answered informationally in public (licensing,
                             catalog / stream-count questions); nothing to
                             verify, nothing actionable to change.
  feature_request_feedback-> acknowledged + "passed to the team" (or pointed
                             at the community ideas board); never resolved,
                             never needs verification.
  general_complaint_vague -> not enough information to act on yet; the real
                             agents ask a clarifying question rather than
                             guessing.
  praise_smalltalk_closed -> no action needed at all (thanks, banter, "it's
                             fixed now").

A `weak_label` keyword/regex heuristic gives a first-pass label for the ~38k
row corpus (used only as *silver* training data / retrieval metadata -- never
as ground truth). Ground truth lives only in eval/golden_set.csv, which was
hand-labeled by a human reviewer using this same taxonomy (see
eval/labeling_notes.md for the process and inter-rater spot-check).
"""
import re

INTENTS = [
    "account_billing",
    "technical_playback",
    "content_catalog",
    "feature_request_feedback",
    "general_complaint_vague",
    "praise_smalltalk_closed",
]

# Order matters: earlier patterns win on overlap (e.g. "cancel my premium"
# should be account_billing even though "premium" alone is not diagnostic).
_PATTERNS = [
    ("account_billing", re.compile(
        r"\b(password|log ?in|login|log out|can'?t sign in|reset my|"
        r"account (was )?hack|hacked|compromised|unauthorized|"
        r"charge(d)?|billing|invoice|refund|payment|paypal|credit card|"
        r"cancel (my )?(premium|subscription|account)|subscription|"
        r"family plan|student (discount|plan|account)|premium code|"
        r"redeem|free trial|upgrade my|downgrade)\b", re.I)),
    ("technical_playback", re.compile(
        r"\b(crash(es|ed|ing)?|won'?t (play|open|load)|keeps? (skipping|"
        r"stopping|pausing|buffering)|buffer(ing)?|freeze(s|d)?|frozen|"
        r"crackl(e|ing)|glitch|lag(ging)?|not (playing|working|loading)|"
        r"error message|bug|offline (mode|download)|sync(ing)?|"
        r"connect(ing)? to|bluetooth|chromecast|cast(ing)? to|shuffle|"
        r"skip(s|ping)?|volume|audio quality|sound quality|app (keeps|is)|"
        r"reinstall(ed)?|update(d)? (the )?app|ios \d|android \d|version \d)\b",
        re.I)),
    ("content_catalog", re.compile(
        r"\b(available in my country|not available|licens(e|ing)|"
        r"remove(d)? from spotify|taken down|missing (album|song|artist|"
        r"track)|stream count|how many streams|explicit version|censored|"
        r"clean version|why (isn'?t|can'?t i find)|catalog|discography)\b",
        re.I)),
    ("feature_request_feedback", re.compile(
        r"\b(feature request|please add|would be (great|nice|awesome|cool) "
        r"if|wish (spotify|you) (had|would)|suggestion|idea|you should "
        r"(add|make|let)|why (don'?t|doesn'?t) spotify|community\.spotify)\b",
        re.I)),
    ("praise_smalltalk_closed", re.compile(
        r"\b(thank(s| you)|thx|appreciate it|solved|fixed( it)?|works? now|"
        r"all (good|sorted|set)|nvm|never mind|love (spotify|this|your)|"
        r"favou?rite (song|artist|album))\b", re.I)),
]

_COMPLAINT_HINTS = re.compile(
    r"\b(fix (this|it)|sucks?|worst|terrible|ridiculous|so (annoying|mad|"
    r"frustrat\w+)|why (does|is) (this|it|spotify)|come on|seriously)\b",
    re.I,
)


def weak_label(text: str) -> str:
    """Fast heuristic label used for silver training data / retrieval
    metadata. NOT used as evaluation ground truth."""
    if not isinstance(text, str) or not text.strip():
        return "general_complaint_vague"
    for intent, pat in _PATTERNS:
        if pat.search(text):
            return intent
    if _COMPLAINT_HINTS.search(text):
        return "general_complaint_vague"
    return "general_complaint_vague"  # default bucket for the unclassifiable
