"""Tests for the model factory: the right class and settings for each provider.

These only construct model objects; they never send a request, so they need no
API keys, AWS login, or running Ollama.

    uv run pytest tests/test_llm.py
"""

from __future__ import annotations

import pytest

from ml_debug_agent.agents import llm


@pytest.fixture(autouse=True)
def fake_aws_credentials(monkeypatch):
    """Construct the Bedrock client the same way on every machine.

    Without this, the tests would depend on the local AWS setup: a profile from .env that
    doesn't exist (e.g. in a container) makes construction fail. Fake keys are never used,
    because these tests don't send requests.
    """
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


def test_bedrock_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    model = llm.get_llm()
    assert type(model).__name__ == "ChatBedrockConverse"
    assert model.model_id == "us.amazon.nova-2-lite-v1:0"
    assert model.region_name == "us-east-1"
    assert model.temperature == 0
    assert not model.additional_model_request_fields  # thinking off by default


def test_bedrock_thinking_and_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    model = llm.get_llm(thinking=True)
    assert model.model_id == "us.amazon.nova-lite-v1:0"
    reasoning = model.additional_model_request_fields["reasoningConfig"]
    assert reasoning == {"type": "enabled", "maxReasoningEffort": "medium"}


def test_provider_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Bedrock")
    assert llm.provider() == "bedrock"


def test_unknown_provider_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="ollama, gemini, bedrock"):
        llm.get_llm()