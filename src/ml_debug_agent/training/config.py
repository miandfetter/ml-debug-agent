"""Bug types and training configs, kept free of PyTorch.

Everything that only needs to know *what* a run is (the bug labels, the config fields,
which fields are hidden from agents) lives here, so the agents and the eval harness can
import it without installing PyTorch. train.py adds the actual training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from dotenv import load_dotenv

DEFAULT_EXPERIMENT = "ml-debug-benchmark"

# Settings like AWS_PROFILE and MLFLOW_ARTIFACT_ROOT come from the project's .env file.
load_dotenv()


class Bug(StrEnum):
    NONE = "none"
    LR_TOO_HIGH = "lr_too_high"
    OVERFIT = "overfit"
    LABEL_SHUFFLE = "label_shuffle"
    LEAKAGE = "leakage"
    UNNORMALIZED = "unnormalized"


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.2
    weight_decay: float = 0.0
    train_subset: int | None = None  # None = use the full training split
    normalize: bool = True
    seed: int = 0
    val_size: int = 10_000
    # Data-pipeline faults. These are NOT logged as params, so the agent has to
    # find them from evidence (curves, split indices) rather than reading the config.
    label_noise: float = 0.0
    leak_val_into_train: bool = False


HIDDEN_FIELDS = {"label_noise", "leak_val_into_train"}

# Each bug is just a set of config changes from the clean baseline.
BUG_PRESETS: dict[Bug, dict[str, Any]] = {
    Bug.NONE: {},
    Bug.LR_TOO_HIGH: {"lr": 0.5},
    Bug.OVERFIT: {"train_subset": 500, "dropout": 0.0, "weight_decay": 0.0, "epochs": 30},
    Bug.LABEL_SHUFFLE: {"label_noise": 1.0},
    Bug.LEAKAGE: {"leak_val_into_train": True},
    Bug.UNNORMALIZED: {"normalize": False},
}


# Human-readable descriptions shown in the MLflow UI (the run's "Description" field).
BUG_DESCRIPTIONS: dict[Bug, str] = {
    Bug.NONE: "Clean baseline: no injected bug.",
    Bug.LR_TOO_HIGH: "Learning rate set to 0.5 (500x baseline). Expect loss explosion, "
    "accuracy stuck near chance, very large gradient norms.",
    Bug.OVERFIT: "Trained on only 500 examples with no dropout or weight decay for 30 epochs. "
    "Expect train accuracy near 100%, val accuracy plateauing low, val loss rising.",
    Bug.LABEL_SHUFFLE: "Training labels randomly permuted (data-pipeline bug, not visible in "
    "params). Expect loss flat near 2.3 and chance-level accuracy.",
    Bug.LEAKAGE: "All validation examples copied into the training set (data-pipeline bug, "
    "not visible in params). Expect inflated val accuracy and test accuracy below val.",
    Bug.UNNORMALIZED: "Inputs left as raw 0-255 pixels instead of normalized. Expect high "
    "initial loss, large early gradients, worse accuracy than clean runs.",
}


def default_run_name(bug: Bug, seed: int) -> str:
    return f"{bug.value}-s{seed}"


def make_config(bug: Bug = Bug.NONE, **overrides: Any) -> TrainConfig:
    """Clean baseline -> apply the bug preset -> apply explicit overrides (e.g. an agent's fix)."""
    cfg = replace(TrainConfig(), **BUG_PRESETS[bug])
    return replace(cfg, **overrides)
