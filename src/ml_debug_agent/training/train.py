"""Train a small MLP on Fashion-MNIST, optionally with an injected bug, and log the run to MLflow.

Usage:
    uv run python -m ml_debug_agent.training.train
    uv run python -m ml_debug_agent.training.train --bug lr_too_high --seed 1
    uv run python -m ml_debug_agent.training.train --bug overfit --dropout 0.3   # preset + override
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import mlflow
import torch
from dotenv import load_dotenv
from torch import nn
from torchvision import datasets

FMNIST_MEAN = 0.2860
FMNIST_STD = 0.3530
DATA_DIR = Path("data")
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


# ---------------------------------------------------------------- data


def _load(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True)
    return ds.data.float(), ds.targets.clone()


def prepare_data(cfg: TrainConfig, gen: torch.Generator) -> dict[str, Any]:
    x_all, y_all = _load(train=True)
    x_test, y_test = _load(train=False)

    perm = torch.randperm(len(x_all), generator=gen)
    val_idx = perm[: cfg.val_size]
    train_idx = perm[cfg.val_size :]

    if cfg.train_subset is not None:
        train_idx = train_idx[: cfg.train_subset]
    if cfg.leak_val_into_train:
        train_idx = torch.cat([train_idx, val_idx])

    x_train, y_train = x_all[train_idx], y_all[train_idx].clone()
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    if cfg.label_noise > 0:
        n = int(cfg.label_noise * len(y_train))
        which = torch.randperm(len(y_train), generator=gen)[:n]
        y_train[which] = y_train[which][torch.randperm(n, generator=gen)]

    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(len(x), -1)
        if cfg.normalize:
            x = (x / 255.0 - FMNIST_MEAN) / FMNIST_STD
        return x  # unnormalized = raw 0-255 pixel values

    return {
        "train": (transform(x_train), y_train),
        "val": (transform(x_val), y_val),
        "test": (transform(x_test), y_test),
        "split": {"train_indices": train_idx.tolist(), "val_indices": val_idx.tolist()},
    }


# ---------------------------------------------------------------- model


def build_model(cfg: TrainConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(28 * 28, cfg.hidden1),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden1, cfg.hidden2),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden2, 10),
    )


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), 2048):
        logits = model(x[i : i + 2048])
        total_loss += loss_fn(logits, y[i : i + 2048]).item()
        correct += (logits.argmax(1) == y[i : i + 2048]).sum().item()
    return total_loss / len(x), correct / len(x)


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if math.isfinite(v)}


# ---------------------------------------------------------------- training


def use_experiment(name: str) -> None:
    """Make `name` the active MLflow experiment, creating it first if it doesn't exist.

    If MLFLOW_ARTIFACT_ROOT is set (e.g. s3://my-bucket/mlflow), a NEW experiment stores
    its artifacts under <root>/<name>. MLflow fixes the location when an experiment is
    created, so existing experiments keep theirs, wherever that is.
    """
    root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if root and mlflow.get_experiment_by_name(name) is None:
        location = f"{root.rstrip('/')}/{name}"
        mlflow.create_experiment(name, artifact_location=location)
        print(f"Created experiment {name!r} with artifacts in {location}")
    mlflow.set_experiment(name)


def train(
    cfg: TrainConfig,
    bug: Bug = Bug.NONE,
    experiment: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
) -> str:
    """Train one run and log it to MLflow. Returns the MLflow run ID."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    data = prepare_data(cfg, gen)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    use_experiment(experiment)
    with mlflow.start_run(run_name=run_name or default_run_name(bug, cfg.seed)) as run:
        mlflow.log_params({k: v for k, v in asdict(cfg).items() if k not in HIDDEN_FIELDS})
        mlflow.log_param("n_train_examples", len(x_train))
        # Ground truth for evaluation. The run name, description, and "eval." tags all reveal
        # the bug, so agent tools must only ever expose params, metrics, and artifacts.
        mlflow.set_tag("eval.true_bug", bug.value)
        mlflow.set_tag("mlflow.note.content", BUG_DESCRIPTIONS[bug])
        mlflow.log_dict(data["split"], "data/split_indices.json")

        status = "completed"
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = torch.randperm(len(x_train), generator=gen)
            loss_sum, correct, seen = 0.0, 0, 0
            grad_norms: list[float] = []

            for i in range(0, len(order), cfg.batch_size):
                idx = order[i : i + cfg.batch_size]
                xb, yb = x_train[idx], y_train[idx]

                logits = model(xb)
                loss = loss_fn(logits, yb)
                if not torch.isfinite(loss):
                    status = "diverged"
                    break

                optimizer.zero_grad()
                loss.backward()
                # max_norm=inf: measures the total gradient norm without clipping it.
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                grad_norms.append(total_norm.item())
                optimizer.step()

                loss_sum += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                seen += len(yb)

            if status == "diverged":
                mlflow.log_metric("diverged_at_epoch", epoch)
                break

            val_loss, val_acc = evaluate(model, x_val, y_val)
            metrics = {
                "train_loss": loss_sum / seen,
                "train_acc": correct / seen,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                "grad_norm_max": max(grad_norms),
                "lr": optimizer.param_groups[0]["lr"],
            }
            mlflow.log_metrics(_finite(metrics), step=epoch)
            print(
                f"epoch {epoch:>2}  train_loss {metrics['train_loss']:.4f}  "
                f"train_acc {metrics['train_acc']:.3f}  "
                f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}"
            )

        if status == "completed":
            test_loss, test_acc = evaluate(model, x_test, y_test)
            mlflow.log_metrics(_finite({"test_loss": test_loss, "test_acc": test_acc}))

        mlflow.set_tag("status", status)
        return run.info.run_id


# ---------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description="Train an MLP on Fashion-MNIST with optional bugs.")
    p.add_argument("--bug", type=Bug, choices=list(Bug), default=Bug.NONE)
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--train-subset", type=int)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--run-name")
    args = p.parse_args()

    overrides = {
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "train_subset": args.train_subset,
    }
    cfg = make_config(args.bug, **{k: v for k, v in overrides.items() if v is not None})
    run_id = train(cfg, bug=args.bug, experiment=args.experiment, run_name=args.run_name)
    print(f"\nRun {run_id} finished (bug={args.bug.value})")


if __name__ == "__main__":
    main()