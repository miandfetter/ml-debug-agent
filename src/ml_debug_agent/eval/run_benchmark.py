"""Evaluate a diagnoser on the labeled benchmark: accuracy, per-bug results, confusion matrix.

This is the ONLY code allowed to read the ground truth (via get_true_bug).

    uv run python -m ml_debug_agent.eval.run_benchmark --system baseline
    uv run python -m ml_debug_agent.eval.run_benchmark --system single_agent --per-bug 1
    uv run python -m ml_debug_agent.eval.run_benchmark --system two_agent --bugs label_shuffle
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ml_debug_agent.agents.mlflow_access import (
    benchmark_experiment,
    get_true_bug,
    list_benchmark_runs,
)
from ml_debug_agent.agents.schemas import BugType, Diagnosis

# Each system is "module:function" for a function taking a run ID and returning a Diagnosis.
# Imported only when chosen, so evaluating the baseline never loads LLM libraries.
SYSTEMS = {
    "baseline": "ml_debug_agent.eval.baseline:diagnose",
    "single_agent": "ml_debug_agent.agents.single_agent:diagnose",
    "single_agent_thinking": "ml_debug_agent.agents.single_agent:diagnose_thinking",
    "two_agent": "ml_debug_agent.agents.graph:diagnose",
}
LABELS = [b.value for b in BugType]  # fixed row/column order for the confusion matrix


def load_system(name: str) -> Callable[[str], Diagnosis]:
    module_name, func_name = SYSTEMS[name].split(":")
    return getattr(importlib.import_module(module_name), func_name)


def select_runs(
    run_ids: list[str], per_bug: int | None = None, bugs: list[str] | None = None
) -> list[str]:
    """Keep only runs of the given bug types, and at most `per_bug` of each. None keeps all."""
    if per_bug is None and bugs is None:
        return run_ids
    taken: dict[str, int] = {}
    kept = []
    for run_id in run_ids:
        bug = get_true_bug(run_id).value
        if bugs is not None and bug not in bugs:
            continue
        if per_bug is None or taken.get(bug, 0) < per_bug:
            taken[bug] = taken.get(bug, 0) + 1
            kept.append(run_id)
    return kept


def evaluate(
    diagnose: Callable[[str], Diagnosis],
    experiment: str,
    per_bug: int | None = None,
    bugs: list[str] | None = None,
) -> pd.DataFrame:
    """Diagnose every selected run (see select_runs) and record prediction vs. truth."""
    rows = []
    run_ids = select_runs(list_benchmark_runs(experiment), per_bug, bugs)
    for i, run_id in enumerate(run_ids, 1):
        truth = get_true_bug(run_id).value
        start = time.time()
        try:
            d = diagnose(run_id)
            predicted, confidence, evidence, error = d.bug.value, d.confidence, d.evidence, ""
        except Exception as e:  # one failed run shouldn't kill the whole eval
            predicted, confidence, evidence, error = "error", None, [], f"{type(e).__name__}: {e}"
        seconds = time.time() - start
        rows.append(
            {
                "run_id": run_id,
                "true_bug": truth,
                "predicted_bug": predicted,
                "correct": predicted == truth,
                "confidence": confidence,
                "seconds": round(seconds, 2),
                "evidence": " | ".join(evidence),
                "error": error,
            }
        )
        mark = "ok " if predicted == truth else "MISS"
        print(f"[{i:>2}/{len(run_ids)}] {mark} true={truth:<14} predicted={predicted}")
        if error:
            print(f"        {error[:500]}")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """Headline metrics from a results table."""
    clean = results[results.true_bug == BugType.NONE.value]
    return {
        "n_runs": len(results),
        "accuracy": results.correct.mean(),
        "per_bug_accuracy": results.groupby("true_bug").correct.mean().reindex(LABELS),
        # Share of clean runs wrongly reported as having a bug.
        "false_positive_rate": (clean.predicted_bug != BugType.NONE.value).mean()
        if len(clean)
        else float("nan"),
        "errors": int((results.predicted_bug == "error").sum()),
        "mean_seconds": results.seconds.mean(),
    }


def confusion_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Rows are the true bug, columns the prediction. The diagonal is correct answers."""
    columns = LABELS + (["error"] if (results.predicted_bug == "error").any() else [])
    return (
        pd.crosstab(results.true_bug, results.predicted_bug)
        .reindex(index=LABELS, columns=columns, fill_value=0)
        .rename_axis(index="true \\ predicted", columns=None)
    )


def results_path(
    out_dir: str | Path,
    experiment: str,
    system: str,
    per_bug: int | None = None,
    bugs: list[str] | None = None,
) -> Path:
    """results/<experiment>/<system>[_per_bugN][_bug-bug].csv

    One folder per benchmark, so results from different benchmarks never overwrite each
    other, and subset runs get their own file so they never overwrite a full run.
    """
    suffix = f"_per_bug{per_bug}" if per_bug else ""
    if bugs:
        suffix += "_" + "-".join(sorted(set(bugs)))
    return Path(out_dir) / experiment / f"{system}{suffix}.csv"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a diagnoser on the benchmark.")
    p.add_argument("--system", choices=sorted(SYSTEMS), default="baseline")
    p.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment to evaluate. Default: BENCHMARK_EXPERIMENT in .env, else v1.",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--per-bug",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N runs of each bug type (a cheap smoke test).",
    )
    p.add_argument(
        "--bugs",
        nargs="+",
        choices=LABELS,
        default=None,
        metavar="BUG",
        help=f"Only evaluate runs of these bug types. Choices: {', '.join(LABELS)}.",
    )
    args = p.parse_args(argv)
    if args.per_bug is not None and args.per_bug < 1:
        p.error("--per-bug must be at least 1")
    args.experiment = args.experiment or benchmark_experiment()

    results = evaluate(load_system(args.system), args.experiment, args.per_bug, args.bugs)

    csv_path = results_path(args.out_dir, args.experiment, args.system, args.per_bug, args.bugs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)

    m = summarize(results)
    print(f"\n=== {args.system} on {args.experiment} ({m['n_runs']} runs) ===")
    print(f"Accuracy:            {m['accuracy']:.1%}")
    fpr = m["false_positive_rate"]
    fpr_text = f"{fpr:.1%}" if pd.notna(fpr) else "n/a (no clean runs evaluated)"
    print(f"False positive rate: {fpr_text}  (clean runs flagged as buggy)")
    print(f"Errors:              {m['errors']}")
    print(f"Mean time per run:   {m['mean_seconds']:.2f}s")
    print("\nPer-bug accuracy:")
    for bug, acc in m["per_bug_accuracy"].items():
        print(f"  {bug:<14} {acc:.0%}" if pd.notna(acc) else f"  {bug:<14} (no runs)")
    print("\nConfusion matrix:")
    print(confusion_matrix(results).to_string())
    print(f"\nPer-run results saved to {csv_path}")


if __name__ == "__main__":
    main()