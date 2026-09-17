import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from intents import weak_label, INTENTS  # noqa: E402


def test_all_intents_reachable():
    assert len(INTENTS) == 6


def test_billing_keywords():
    assert weak_label("I want to cancel my premium subscription") == "account_billing"


def test_technical_keywords():
    assert weak_label("the app keeps crashing and buffering on my phone") == "technical_playback"


def test_catalog_keywords():
    assert weak_label("this album is not available in my country") == "content_catalog"


def test_feature_request_keywords():
    assert weak_label("please add a sleep timer feature") == "feature_request_feedback"


def test_praise_keywords():
    assert weak_label("thanks so much, it's fixed now!") == "praise_smalltalk_closed"


def test_empty_text_defaults_to_vague():
    assert weak_label("") == "general_complaint_vague"
    assert weak_label(None) == "general_complaint_vague"


def test_unmatched_text_defaults_to_vague():
    assert weak_label("hello there") == "general_complaint_vague"
