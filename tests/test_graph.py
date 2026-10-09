"""Tests for the two-agent graph's plumbing, using scripted fake models instead of an LLM.

These check routing, the follow-up loop and its cap, and how evidence and decisions are
passed between the agents. They don't test diagnosis quality; that's what the eval is for.

    uv run pytest
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ml_debug_agent.agents import graph
from ml_debug_agent.agents.schemas import BugType, Diagnosis
from ml_debug_agent.training.train import Bug


class ScriptedInvestigator(GenericFakeChatModel):
    """Returns pre-written replies in order. Accepts tools but ignores them."""

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedDiagnoser:
    """Returns pre-written reasoning in order and records the system prompt of each turn."""

    def __init__(self, replies: list[str], extracted_bug: str = "none"):
        self.replies = iter(replies)
        self.system_prompts: list[str] = []
        self.extracted_bug = extracted_bug

    def invoke(self, messages):
        self.system_prompts.append(messages[0].content)
        return AIMessage(next(self.replies))

    def with_structured_output(self, schema):
        assert schema is Diagnosis
        fixed = Diagnosis(evidence=["from the reasoning"], bug=self.extracted_bug, confidence=0.7)
        return RunnableLambda(lambda _: fixed)


def _calls(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {"name": name, "args": args, "id": f"{name}-{i}"}
            for i, (name, args) in enumerate(calls)
        ],
    )


ROUND_1 = [
    _calls(("summarize_curves", {}), ("get_config", {}), ("check_split_overlap", {})),
    AIMessage("Done collecting."),
]


# --- parse_decision -----------------------------------------------------------------------


def test_parse_decision_reads_the_closing_line():
    assert graph.parse_decision("Loss is flat.\nFinal answer: overfit") == (BugType.OVERFIT, None)
    assert graph.parse_decision("**Final answer:** `label_shuffle`") == (
        BugType.LABEL_SHUFFLE,
        None,
    )
    assert graph.parse_decision("Hmm.\nNeed more evidence: val_loss per epoch") == (
        None,
        "val_loss per epoch",
    )


def test_parse_decision_uses_the_last_decision_line():
    text = "Need more evidence: the config\n...after thinking more...\nFinal answer: leakage"
    assert graph.parse_decision(text) == (BugType.LEAKAGE, None)


def test_parse_decision_rejects_unknown_labels_and_missing_lines():
    assert graph.parse_decision("Final answer: exploding_gradients") == (None, None)
    assert graph.parse_decision("I am not sure what is wrong.") == (None, None)


# --- investigator -------------------------------------------------------------------------


def test_first_investigation_files_tool_outputs_into_the_report(runs):
    investigator = ScriptedInvestigator(messages=iter(ROUND_1))
    report = graph.first_investigation(runs[Bug.LEAKAGE], investigator)
    assert report.split_overlap["n_overlap"] == 500  # the real tool ran against the real run
    assert "lr" in report.config
    assert "train_loss_final" in report.curve_summary
    assert "Filled in by code" not in report.investigator_notes


def test_first_investigation_fills_in_skipped_tools(runs):
    lazy = ScriptedInvestigator(
        messages=iter([_calls(("summarize_curves", {})), AIMessage("That's enough.")])
    )
    report = graph.first_investigation(runs[Bug.LEAKAGE], lazy)
    assert report.config is not None and report.split_overlap is not None
    assert "Filled in by code: ['get_config', 'check_split_overlap']" in report.investigator_notes


# --- full graph ---------------------------------------------------------------------------


def test_follow_up_request_loops_back_to_the_investigator(runs):
    investigator = ScriptedInvestigator(
        messages=iter(
            [
                *ROUND_1,
                _calls(("get_metric_history", {"metric": "val_loss"})),
                AIMessage("Here is val_loss."),
            ]
        )
    )
    diagnoser = ScriptedDiagnoser(
        ["Need more evidence: val_loss at every epoch", "Final answer: leakage"]
    )
    state = graph.run_graph(runs[Bug.LEAKAGE], investigator, diagnoser)

    assert state["rounds"] == 1
    assert state["diagnosis"].bug is BugType.LEAKAGE
    assert len(state["diagnoser_reasoning"]) == 2
    [finding] = state["evidence"].extra_findings
    assert finding.startswith('get_metric_history({"metric": "val_loss"})')


def test_follow_ups_are_capped_and_the_diagnoser_must_commit(runs):
    follow_up_round = [_calls(("get_metric_history", {"metric": "val_acc"})), AIMessage("ok")]
    investigator = ScriptedInvestigator(
        messages=iter([*ROUND_1, *follow_up_round, *follow_up_round])
    )
    # Keeps asking, even on the last turn when it's told to commit.
    diagnoser = ScriptedDiagnoser(["Need more evidence: more"] * 3, extracted_bug="overfit")
    state = graph.run_graph(runs[Bug.OVERFIT], investigator, diagnoser)

    assert state["rounds"] == graph.MAX_FOLLOW_UPS
    assert state["diagnosis"].bug is BugType.OVERFIT  # extraction decides when no label given
    assert [graph.MUST_COMMIT in p for p in diagnoser.system_prompts] == [False, False, True]


def test_free_text_label_overrides_the_extraction(runs):
    investigator = ScriptedInvestigator(messages=iter(ROUND_1))
    diagnoser = ScriptedDiagnoser(["Final answer: label_shuffle"], extracted_bug="none")
    state = graph.run_graph(runs[Bug.LABEL_SHUFFLE], investigator, diagnoser)
    assert state["diagnosis"].bug is BugType.LABEL_SHUFFLE
    assert state["rounds"] == 0
