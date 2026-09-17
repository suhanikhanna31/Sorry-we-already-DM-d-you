import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from draft import draft_reply  # noqa: E402


def test_empty_retrieval_returns_safe_fallback():
    out = draft_reply("technical_playback", [])
    assert out["grounded_in"] is None
    assert out["similarity"] == 0.0
    assert len(out["draft"]) > 0


def test_strips_signature_and_sentence_initial_greeting():
    results = [{
        "similarity": 0.9,
        "customer_text": "my app keeps crashing on android",
        "agent_reply": "Hey Emily! Try logging out and back in. /JI",
        "is_dm_only": False,
    }]
    out = draft_reply("technical_playback", results)
    assert "Emily" not in out["draft"]
    assert "/JI" not in out["draft"]
    assert "Try logging out and back in." in out["draft"]


def test_prefers_substantive_over_dm_only_boilerplate():
    results = [
        {
            "similarity": 0.95,
            "customer_text": "a",
            "agent_reply": "We've just sent you a DM!",
            "is_dm_only": True,
        },
        {
            "similarity": 0.80,
            "customer_text": "b",
            "agent_reply": "Try restarting the app and let us know if that helps.",
            "is_dm_only": False,
        },
    ]
    out = draft_reply("technical_playback", results)
    assert "restarting" in out["draft"]


def test_falls_back_to_top_candidate_when_all_are_dm_only():
    results = [{
        "similarity": 0.9,
        "customer_text": "a",
        "agent_reply": "We've just sent you a DM!",
        "is_dm_only": True,
    }]
    out = draft_reply("account_billing", results)
    assert out["grounded_in"] == "a"
