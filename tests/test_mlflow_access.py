"""Tests for the MLflow gatekeeper, above all that the agent's view never reveals the answer.

Each test trains real (tiny) runs through train() on random stand-in data, logged to a
temporary MLflow store, so it exercises the exact logging path the benchmark uses
without downloading Fashion-MNIST.

    uv run pytest
"""

from __future__ import annotations

import mlflow
import pytest
import torch

import ml_debug_agent.training.train as train_mod
from ml_debug_agent.agents.mlflow_access import (
    get_run_view,
    get_split_indices,
    get_true_bug,
    list_benchmark_runs,
)
from ml_debug_agent.training.train import BUG_DESCRIPTIONS, HIDDEN_FIELDS, Bug, make_config

EXPERIMENT = "test-benchmark"
BUGS = [bug for bug in Bug if bug is not Bug.NONE]


def _fake_load(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    n = 3000 if train else 500
    g = torch.Generator().manual_seed(1 if train else 2)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.rand(n, 28, 28, generator=g) * 60 + y.view(-1, 1, 1).float() * 15
    return x, y


@pytest.fixture
def runs(tmp_path, monkeypatch) -> dict[Bug, str]:
    """One tiny run per bug type (including clean), in a throwaway MLflow store."""
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo
    monkeypatch.setattr(train_mod, "_load", _fake_load)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test.db'}")
    ids = {}
    for bug in Bug:
        cfg = make_config(bug, epochs=2, val_size=500)
        ids[bug] = train_mod.train(cfg, bug=bug, experiment=EXPERIMENT)
    yield ids
    mlflow.set_tracking_uri(None)


@pytest.mark.parametrize("bug", BUGS, ids=[b.value for b in BUGS])
def test_view_never_reveals_the_answer(runs, bug):
    run_id = runs[bug]
    view_text = get_run_view(run_id).model_dump_json()
    run = mlflow.get_run(run_id)

    assert bug.value not in view_text, "bug name leaked"
    assert run.info.run_name not in view_text, "run name leaked"
    assert BUG_DESCRIPTIONS[bug] not in view_text, "run description leaked"
    assert "eval." not in view_text, "an eval tag leaked"
    for hidden in HIDDEN_FIELDS:
        assert hidden not in view_text, f"hidden field {hidden!r} leaked"


def test_view_contains_what_the_agent_needs(runs):
    view = get_run_view(runs[Bug.LR_TOO_HIGH])
    assert view.params["lr"] == 0.5  # parsed back to a float, not the string "0.5"
    assert view.params["normalize"] is True
    assert len(view.history["val_acc"]) == 2  # one value per epoch
    assert "test_acc" in view.final_metrics
    assert view.training_status in {"completed", "diverged"}


def test_split_overlap_only_in_leakage_run(runs):
    def overlap(run_id: str) -> int:
        split = get_split_indices(run_id)
        return len(set(split["train_indices"]) & set(split["val_indices"]))

    assert overlap(runs[Bug.LEAKAGE]) > 0
    assert overlap(runs[Bug.NONE]) == 0


def test_true_bug_and_listing(runs):
    assert set(list_benchmark_runs(EXPERIMENT)) == set(runs.values())
    for bug, run_id in runs.items():
        assert get_true_bug(run_id) is bug