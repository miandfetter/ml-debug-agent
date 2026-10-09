"""Tests for the investigator's tools: correct features, working LangChain wrappers, no leaks.

    uv run pytest
"""

from __future__ import annotations

import json

import mlflow
import pytest

from ml_debug_agent.agents.tools import (
    CHANCE_LOSS,
    build_tools,
    curve_summary,
    metric_history,
    split_overlap,
)
from ml_debug_agent.training.train import BUG_DESCRIPTIONS, Bug

EXPECTED_TOOLS = {"get_config", "get_metric_history", "summarize_curves", "check_split_overlap"}
BUGS = [bug for bug in Bug if bug is not Bug.NONE]


def _call_all(run_id: str) -> dict[str, dict]:
    """Invoke every tool the way the agent would, and parse the JSON each returns."""
    out = {}
    for t in build_tools(run_id):
        args = {"metric": "val_loss"} if t.name == "get_metric_history" else {}
        out[t.name] = json.loads(t.invoke(args))
    return out


def test_tools_are_built_and_callable(runs):
    tools = build_tools(runs[Bug.NONE])
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description for t in tools)  # docstrings become what the LLM reads
    results = _call_all(runs[Bug.NONE])
    assert results["get_config"]["lr"] == 0.001
    assert len(results["get_metric_history"]["values"]) == 3


@pytest.mark.parametrize("bug", BUGS, ids=[b.value for b in BUGS])
def test_tool_outputs_never_reveal_the_answer(runs, bug):
    text = json.dumps(_call_all(runs[bug]))
    assert bug.value not in text
    assert mlflow.get_run(runs[bug]).info.run_name not in text
    assert BUG_DESCRIPTIONS[bug] not in text


def test_unknown_metric_returns_helpful_error(runs):
    result = metric_history(runs[Bug.NONE], "accuracy")
    assert "error" in result
    assert "val_acc" in result["available"]  # tells the model what it can ask for instead


def test_curve_summary_has_every_feature_for_every_run(runs):
    expected_keys = set(curve_summary(runs[Bug.NONE]))
    for run_id in runs.values():
        assert set(curve_summary(run_id)) == expected_keys


def test_curve_summary_distinguishes_bugs(runs):
    clean = curve_summary(runs[Bug.NONE])
    lr_high = curve_summary(runs[Bug.LR_TOO_HIGH])
    shuffled = curve_summary(runs[Bug.LABEL_SHUFFLE])

    assert clean["chance_level_loss"] == CHANCE_LOSS
    # A huge learning rate blows up the early loss.
    assert lr_high["train_loss_first_epoch"] > 2 * clean["train_loss_first_epoch"]
    # Shuffled labels leave nothing to learn: loss stays near chance, clean loss drops below it.
    assert abs(shuffled["train_loss_final"] - CHANCE_LOSS) < 0.2
    assert clean["train_loss_final"] < shuffled["train_loss_final"]


def test_split_overlap_tool(runs):
    leak, clean = split_overlap(runs[Bug.LEAKAGE]), split_overlap(runs[Bug.NONE])
    assert leak["n_overlap"] == leak["n_val"]  # the whole validation set was copied in
    assert leak["overlap_fraction_of_val"] == 1.0
    assert clean["n_overlap"] == 0