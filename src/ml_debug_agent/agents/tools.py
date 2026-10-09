"""Tools the investigator agent uses to gather evidence about a training run."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from langchain_core.tools import BaseTool, tool

from ml_debug_agent.agents.mlflow_access import (
    get_run_view,
    get_split_indices,
    list_benchmark_runs,
)

# Reference points for a 10-class problem, included so the agent doesn't have to know them.
CHANCE_LOSS = round(math.log(10), 4)  # cross-entropy of a uniform guess over 10 classes
CHANCE_ACC = 0.1


def _r(x: float | None) -> float | None:
    """Round for readability. Small models handle 0.4127 better than 0.41269874572753906."""
    return None if x is None else round(float(x), 4)


def run_config(run_id: str) -> dict[str, Any]:
    """The run's logged hyperparameters."""
    return get_run_view(run_id).params


def metric_history(run_id: str, metric: str) -> dict[str, Any]:
    """One metric's value at every epoch."""
    history = get_run_view(run_id).history
    if metric not in history:
        return {"error": f"No metric {metric!r} for this run.", "available": sorted(history)}
    values = history[metric]
    return {
        "metric": metric,
        "epochs": list(range(1, len(values) + 1)),
        "values": [_r(v) for v in values],
    }

def curve_summary(run_id: str) -> dict[str, Any]:
    """Numeric features of the training curves, computed so the LLM doesn't do arithmetic.

    Raw measurements only: no interpretation or bug names. Deciding what the numbers
    mean is the diagnoser's job.
    """
    view = get_run_view(run_id)
    h, final = view.history, view.final_metrics
    train_loss, val_loss = h.get("train_loss", []), h.get("val_loss", [])
    train_acc, val_acc = h.get("train_acc", []), h.get("val_acc", [])
    grad_max, grad_mean = h.get("grad_norm_max", []), h.get("grad_norm_mean", [])
    test_acc = final.get("test_acc")

    def last(xs: list[float]) -> float | None:
        return xs[-1] if xs else None

    min_val_loss = min(val_loss) if val_loss else None

    return {
        # How the run ended
        "training_status": view.training_status,
        "epochs_completed": len(val_loss),
        "epochs_planned": view.params.get("epochs"),
        "diverged_at_epoch": final.get("diverged_at_epoch"),
        # Reference values for this 10-class task
        "chance_level_loss": CHANCE_LOSS,
        "chance_level_acc": CHANCE_ACC,
        # Loss
        "train_loss_first_epoch": _r(train_loss[0]) if train_loss else None,
        "train_loss_final": _r(last(train_loss)),
        "val_loss_final": _r(last(val_loss)),
        "val_loss_min": _r(min_val_loss),
        "val_loss_min_epoch": val_loss.index(min_val_loss) + 1 if val_loss else None,
        "val_loss_rise_since_min": _r(val_loss[-1] - min_val_loss) if val_loss else None,
        "epochs_where_val_loss_increased": sum(
            b > a for a, b in zip(val_loss, val_loss[1:], strict=False)
        ),
        # Accuracy
        "train_acc_final": _r(last(train_acc)),
        "val_acc_final": _r(last(val_acc)),
        "val_acc_best": _r(max(val_acc)) if val_acc else None,
        "train_val_acc_gap": _r(train_acc[-1] - val_acc[-1]) if train_acc and val_acc else None,
        "test_acc": _r(test_acc),
        "val_test_acc_gap": _r(val_acc[-1] - test_acc)
        if val_acc and test_acc is not None
        else None,
        # Gradients
        "grad_norm_max_overall": _r(max(grad_max)) if grad_max else None,
        "grad_norm_mean_first_epoch": _r(grad_mean[0]) if grad_mean else None,
        "grad_norm_mean_final_epoch": _r(last(grad_mean)),
    }

def split_overlap(run_id: str) -> dict[str, Any]:
    """How many examples appear in both the training and validation sets."""
    split = get_split_indices(run_id)
    train, val = set(split["train_indices"]), set(split["val_indices"])
    overlap = len(train & val)
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_overlap": overlap,
        "overlap_fraction_of_val": _r(overlap / len(val)) if val else None,
    }

def build_tools(run_id: str) -> list[BaseTool]:
    """LangChain tools bound to one run.

    The run ID is fixed inside each tool rather than passed by the model, so the agent
    can't mistype a 32-character ID or wander off to inspect other runs.
    """

    @tool
    def get_config() -> str:
        """Get the run's hyperparameters: learning rate, batch size, epochs, layer sizes,
        dropout, weight decay, training subset size, whether inputs are normalized, and
        the number of training examples. Use this to check for misconfiguration."""
        return json.dumps(run_config(run_id))

    @tool
    def get_metric_history(metric: str) -> str:
        """Get one metric's value at every epoch, to inspect a curve's shape directly.

        Args:
            metric: One of train_loss, train_acc, val_loss, val_acc, grad_norm_mean,
                grad_norm_max, lr.
        """
        return json.dumps(metric_history(run_id, metric))

    @tool
    def summarize_curves() -> str:
        """Get numeric features of the training curves: first and final losses, the
        difference between train and validation accuracy, whether validation loss rose
        after its minimum, validation vs. test accuracy, gradient norms, and reference
        values for a uniform random guess. Usually the best tool to call first."""
        return json.dumps(curve_summary(run_id))

    @tool
    def check_split_overlap() -> str:
        """Check whether any examples appear in both the training and validation sets."""
        return json.dumps(split_overlap(run_id))

    return [get_config, get_metric_history, summarize_curves, check_split_overlap]


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else list_benchmark_runs()[0]
    for t in build_tools(target):
        args = {"metric": "val_acc"} if t.name == "get_metric_history" else {}
        print(f"\n=== {t.name} ===")
        print(json.dumps(json.loads(t.invoke(args)), indent=2))