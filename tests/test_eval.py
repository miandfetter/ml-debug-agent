"""Tests for the eval harness and the baseline.

    uv run pytest
"""

from __future__ import annotations

import pandas as pd

from ml_debug_agent.agents.schemas import BugType, Diagnosis
from ml_debug_agent.eval import baseline
from ml_debug_agent.eval.run_benchmark import confusion_matrix, evaluate, summarize


def _row(true: str, pred: str) -> dict:
    return {"true_bug": true, "predicted_bug": pred, "correct": true == pred, "seconds": 0.1}


def test_summarize_and_confusion_matrix_on_known_results():
    results = pd.DataFrame(
        [
            _row("none", "none"),
            _row("none", "overfit"),  # a false positive
            _row("overfit", "overfit"),
            _row("leakage", "none"),  # a miss
        ]
    )
    m = summarize(results)
    assert m["accuracy"] == 0.5
    assert m["false_positive_rate"] == 0.5
    assert m["per_bug_accuracy"]["overfit"] == 1.0
    assert m["per_bug_accuracy"]["leakage"] == 0.0

    cm = confusion_matrix(results)
    assert cm.loc["none", "overfit"] == 1
    assert cm.loc["leakage", "none"] == 1
    assert list(cm.index) == [b.value for b in BugType]  # every bug gets a row


def test_evaluate_records_errors_instead_of_crashing(runs):
    def broken(run_id: str) -> Diagnosis:
        raise RuntimeError("model returned garbage")

    results = evaluate(broken, _experiment_of(runs))
    assert (results.predicted_bug == "error").all()
    assert summarize(results)["errors"] == len(runs)


def test_baseline_runs_end_to_end(runs):
    results = evaluate(baseline.diagnose, _experiment_of(runs))
    assert len(results) == len(runs)
    assert (results.predicted_bug != "error").all()
    # Direct evidence should always be caught, even on stand-in data.
    leak = results[results.true_bug == "leakage"]
    assert (leak.predicted_bug == "leakage").all()


def test_evaluate_per_bug_takes_n_runs_of_each_bug(runs):
    results = evaluate(baseline.diagnose, _experiment_of(runs), per_bug=1)
    assert sorted(results.true_bug) == sorted(b.value for b in BugType)


def _experiment_of(runs) -> str:
    import mlflow

    run = mlflow.get_run(next(iter(runs.values())))
    return mlflow.get_experiment(run.info.experiment_id).name