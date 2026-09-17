"""
llm_client.py

Thin, optional wrapper around the Anthropic Messages API. Used only when the
agent or judge is run with --mode llm / for eval/llm_judge.py, and only if
ANTHROPIC_API_KEY is set in the environment.

This is NOT required to reproduce the headline numbers in the README -- the
default pipeline (src/agent.py --mode local) is 100% offline: TF-IDF
retrieval + a trained scikit-learn classifier + rule-based escalation. The
LLM mode exists to show the pluggable "production" path the task description
allows ("you may use any LLM API"), and to power the LLM-as-judge rubric
scorer, which inherently needs an LLM.
"""
import json
import os
import urllib.request

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


def call(system: str, user: str, max_tokens: int = 1000) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. LLM mode / LLM judge requires an API "
            "key -- see README 'Optional: LLM mode'. The local pipeline "
            "(default) does not need this."
        )
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", []))
