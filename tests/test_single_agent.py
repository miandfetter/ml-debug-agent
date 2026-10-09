"""Tests for the single agent's plumbing, using a scripted fake model instead of Ollama.

These check that tool calls are executed, evidence is collected, and the answer step
returns a Diagnosis. They don't test diagnosis quality; that's what the eval is for.

    uv run pytest
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ml_debug_agent.agents import graph, single_agent
from ml_debug_agent.agents.schemas import BugType, Diagnosis
from ml_debug_agent.agents.tools import build_tools
from ml_debug_agent.training.train import Bug


class ScriptedModel(GenericFakeChatModel):
    """Returns pre-written replies in order. Accepts tools but ignores them."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_call(name: str, call_id: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": {}, "id": call_id}])


def test_investigate_runs_tools_and_collects_evidence(runs):
    model = ScriptedModel(
        messages=iter(
            [
                _tool_call("summarize_curves", "1"),
                _tool_call("check_split_overlap", "2"),
                AIMessage("All 500 validation examples also appear in training."),
            ]
        )
    )
    messages = single_agent.investigate(runs[Bug.LEAKAGE], llm=model)
    evidence = single_agent.format_evidence(messages)

    assert "### summarize_curves" in evidence
    assert "### check_split_overlap" in evidence
    assert '"n_overlap": 500' in evidence  # the real tool ran against the real run
    # The investigator's interpretation is left out by default, available on request.
    assert "### Investigator's summary" not in evidence
    with_summary = single_agent.format_evidence(messages, include_summary=True)
    assert "### Investigator's summary" in with_summary


def test_format_evidence_handles_no_tool_calls():
    assert single_agent.format_evidence([AIMessage("")]) == "No evidence was gathered."


def test_answer_returns_a_diagnosis():
    class FakeStructured:
        def with_structured_output(self, schema):
            assert schema is Diagnosis
            fixed = Diagnosis(evidence=["n_overlap is 500"], bug="leakage", confidence=0.9)
            return RunnableLambda(lambda _: fixed)

    d = single_agent.answer("some evidence", llm=FakeStructured())
    assert d.bug is BugType.LEAKAGE


# Words that would describe a bug's symptoms or thresholds rather than define it.
GIVEAWAYS = ["2.3", "0.5", "gap", "explod", "chance", "nonzero"]


def test_prompts_do_not_contain_symptoms():
    """Keep the comparison with the baseline fair: definitions only, no diagnostic rules."""
    text = "".join(
        [
            single_agent.INVESTIGATE_PROMPT,
            single_agent.ANSWER_PROMPT,
            graph.INVESTIGATOR_PROMPT,
            graph.FOLLOW_UP_PROMPT,
            graph.DIAGNOSER_PROMPT,
            graph.MUST_COMMIT,
            graph.EXTRACT_PROMPT,
        ]
    )
    # Tool names (like check_split_overlap) may appear; symptoms and thresholds may not.
    for giveaway in GIVEAWAYS:
        assert giveaway not in text.lower(), giveaway


def test_tool_descriptions_do_not_contain_symptoms():
    """The model reads tool descriptions too, so they follow the same rule as the prompts."""
    for t in build_tools("any-run-id"):  # descriptions don't depend on the run
        for giveaway in GIVEAWAYS:
            assert giveaway not in t.description.lower(), (t.name, giveaway)