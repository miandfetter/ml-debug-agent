"""Generate the labeled benchmark: every bug type x several seeds, logged to MLflow.

Usage (from the repo root):
    uv run python scripts/generate_benchmark.py
    uv run python scripts/generate_benchmark.py --seeds 3            # quicker, smaller benchmark
    uv run python scripts/generate_benchmark.py --bugs none overfit  # only some bug types

Safe to re-run: runs that already exist in the experiment are skipped, so if it's
interrupted you can just start it again.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace

import mlflow

from ml_debug_agent.agents.mlflow_access import benchmark_experiment
from ml_debug_agent.training.config import BUG_PRESETS, Bug, TrainConfig, default_run_name
from ml_debug_agent.training.train import train, use_experiment

EXPERIMENT_DESCRIPTION = (
    "Labeled benchmark for the ML debugging agent: a small MLP on Fashion-MNIST with "
    "5 injected bug types (lr_too_high, overfit, label_shuffle, leakage, unnormalized) "
    "plus clean controls (none), several seeds each. Run names are '<bug>-s<seed>'; "
    "ground truth is in the 'eval.true_bug' tag."
)

# Small, realistic variation in the healthy settings so runs of the same bug don't
# look identical. The bug preset is applied on top, so it always wins.
VARIATION = {
    "lr": [5e-4, 1e-3, 2e-3],
    "batch_size": [64, 128, 256],
    "hidden1": [128, 256, 512],
    "dropout": [0.1, 0.2, 0.3],
}


def config_for(bug: Bug, seed: int) -> TrainConfig:
    rng = random.Random(f"{bug.value}-{seed}")  # deterministic per (bug, seed)
    variation = {name: rng.choice(options) for name, options in VARIATION.items()}
    cfg = replace(TrainConfig(), seed=seed, **variation)
    return replace(cfg, **BUG_PRESETS[bug])


def existing_run_names(experiment: str) -> set[str]:
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return set()
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    return {r.info.run_name for r in runs if r.info.status == "FINISHED"}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the labeled benchmark of training runs.")
    p.add_argument("--seeds", type=int, default=5, help="runs per bug type")
    p.add_argument("--bugs", nargs="+", type=Bug, choices=list(Bug), default=list(Bug))
    p.add_argument(
        "--experiment",
        default=None,
        help="Experiment to generate into. Default: BENCHMARK_EXPERIMENT in .env, else v1.",
    )
    args = p.parse_args()
    args.experiment = args.experiment or benchmark_experiment()

    use_experiment(args.experiment)
    print(f"Artifacts go to: {mlflow.get_experiment_by_name(args.experiment).artifact_location}")
    mlflow.set_experiment_tag("mlflow.note.content", EXPERIMENT_DESCRIPTION)

    done = existing_run_names(args.experiment)
    jobs = [(bug, seed) for bug in args.bugs for seed in range(args.seeds)]
    todo = [(bug, seed) for bug, seed in jobs if default_run_name(bug, seed) not in done]
    print(
        f"{len(jobs)} runs in benchmark, {len(jobs) - len(todo)} already done, "
        f"{len(todo)} to train\n"
    )

    start = time.time()
    for i, (bug, seed) in enumerate(todo, 1):
        name = default_run_name(bug, seed)
        print(f"[{i}/{len(todo)}] {name}")
        t0 = time.time()
        train(config_for(bug, seed), bug=bug, experiment=args.experiment, run_name=name)
        print(f"    done in {time.time() - t0:.0f}s\n")

    print(f"Finished {len(todo)} runs in {(time.time() - start) / 60:.1f} min "
          f"-> MLflow experiment '{args.experiment}'")


if __name__ == "__main__":
    main()