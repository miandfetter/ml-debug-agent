"""Shared test fixtures. Pytest loads this file automatically for every test in tests/.

The `runs` fixture trains one tiny run per bug type (including clean) on random
stand-in data, in a throwaway MLflow store. It's session-scoped: built once and
shared by every test, so the suite trains 6 runs total instead of 6 per test.
"""

from __future__ import annotations

import mlflow
import pytest
import torch

import ml_debug_agent.training.train as train_mod
from ml_debug_agent.training.train import Bug, make_config

TEST_EXPERIMENT = "test-benchmark"


def _fake_load(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand-in for Fashion-MNIST: pixel brightness depends on the label, so it's learnable."""
    n = 3000 if train else 500
    g = torch.Generator().manual_seed(1 if train else 2)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.rand(n, 28, 28, generator=g) * 60 + y.view(-1, 1, 1).float() * 15
    return x, y


@pytest.fixture(scope="session")
def runs(tmp_path_factory) -> dict[Bug, str]:
    tmp = tmp_path_factory.mktemp("mlflow")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)  # keep MLflow artifacts out of the repo
        mp.setattr(train_mod, "_load", _fake_load)
        mlflow.set_tracking_uri(f"sqlite:///{tmp / 'test.db'}")
        ids = {}
        for bug in Bug:
            cfg = make_config(bug, epochs=3, val_size=500)
            ids[bug] = train_mod.train(cfg, bug=bug, experiment=TEST_EXPERIMENT)
        yield ids
        mlflow.set_tracking_uri(None)