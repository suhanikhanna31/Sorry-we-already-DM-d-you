"""
Unit tests for src/escalation.py -- the rule-based auto-handle/escalate gate.

These run with zero external data (no trained model, no raw dataset), so
they're part of the fast CI smoke test as well as local development.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import escalation  # noqa: E402


def test_security_language_always_escalates():
    should, reason = escalation.decide(
        "someone hacked my account and changed my password", "account_billing", 0.9
    )
    assert should is True
    assert "security" in reason


def test_billing_harm_always_escalates():
    should, reason = escalation.decide(
        "I was charged twice for my subscription this month", "account_billing", 0.9
    )
    assert should is True
    assert "billing" in reason


def test_public_pii_always_escalates():
    should, reason = escalation.decide(
        "my email is user@example.com, please fix my account", "account_billing", 0.9
    )
    assert should is True
    assert "personal info" in reason


def test_low_confidence_escalates():
    should, reason = escalation.decide("some ambiguous message", "content_catalog", 0.1)
    assert should is True
    assert "confidence too low" in reason


def test_short_vague_message_escalates_despite_high_confidence():
    should, reason = escalation.decide("Help, please?", "general_complaint_vague", 0.95)
    assert should is True
    assert "too short" in reason


def test_routine_message_does_not_escalate():
    should, reason = escalation.decide(
        "My song keeps skipping on my iPhone running iOS 17", "technical_playback", 0.9
    )
    assert should is False
    assert "safe-to-automate" in reason


def test_profanity_alone_does_not_escalate():
    # Profanity alone (no long-unresolved / churn signal) should not trip
    # escalation on its own -- see escalation.py's combined-signal logic.
    should, _ = escalation.decide("this app is such shit", "general_complaint_vague", 0.8)
    assert should is False
