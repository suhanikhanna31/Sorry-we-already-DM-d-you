"""
baselines.py

Two baselines, run against the same golden set as the main agent:

TRIVIAL: always predicts the single most common intent in the training
corpus, always gives one fixed generic canned reply, and never escalates
(fully automated, zero human involvement, zero awareness of the message).
This is the "what if we did the laziest possible thing" floor.

SIMPLE: uses the same trained TF-IDF+LogReg classifier as the main agent,
but replies with one fixed canned template per intent (no retrieval, no
personalization, no adaptation) and escalates using one naive rule
("escalate iff predicted intent is account_billing, since that's the only
intent that's ever sensitive-looking on the surface").
"""
import sys
sys.path.insert(0, "src")
from classifier import IntentClassifier

MAJORITY_INTENT = "general_complaint_vague"
TRIVIAL_REPLY = "Hi there! Thanks for reaching out. Can you tell us a bit more about what's going on so we can take a look?"

SIMPLE_TEMPLATES = {
    "account_billing": "Hey there! Can you DM us your account's email address or username? We'll take a look backstage.",
    "technical_playback": "Hey! Can you let us know your device, OS, and Spotify version? We'll see what we can suggest.",
    "content_catalog": "Hey there! Fingers crossed we'll be able to have it soon, but there's more info about Spotify content here.",
    "feature_request_feedback": "Hi! Thanks for the suggestion - we'll pass it on to the right team.",
    "general_complaint_vague": "Hi there! Thanks for reaching out. Can you tell us a bit more about what's going on so we can help?",
    "praise_smalltalk_closed": "Thanks so much! Let us know if there's ever anything else we can help with.",
}


class TrivialBaseline:
    name = "trivial"

    def handle(self, text: str) -> dict:
        return {
            "intent": MAJORITY_INTENT,
            "intent_confidence": None,
            "draft_reply": TRIVIAL_REPLY,
            "should_escalate": False,
            "escalate_reason": "trivial baseline never escalates",
        }


class SimpleBaseline:
    name = "simple"

    def __init__(self):
        self.classifier = IntentClassifier()

    def handle(self, text: str) -> dict:
        intent, confidence = self.classifier.predict(text)
        should_escalate = intent == "account_billing"
        reason = (
            "naive rule: predicted intent is account_billing"
            if should_escalate else "naive rule: predicted intent is not account_billing"
        )
        return {
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "draft_reply": SIMPLE_TEMPLATES[intent],
            "should_escalate": should_escalate,
            "escalate_reason": reason,
        }
