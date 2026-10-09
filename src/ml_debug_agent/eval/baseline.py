"""Rule-based baseline diagnoser: hand-written if-statements over the evidence the agent sees."""

from __future__ import annotations

from ml_debug_agent.agents.schemas import BugType, Diagnosis
from ml_debug_agent.agents.tools import CHANCE_LOSS, curve_summary, run_config, split_overlap

# Thresholds, named so they're easy to find, tune, and discuss in the README.
HIGH_LR = 0.1  # 100x the usual Adam default of 1e-3
EXPLODED_FIRST_LOSS = 5.0  # a healthy first epoch is ~0.5; chance is ~2.3
NEAR_CHANCE_LOSS = 0.15  # final train loss within this distance of log(10)
NEAR_CHANCE_ACC = 0.2  # final val accuracy below this is close to random guessing
OVERFIT_ACC_GAP = 0.08  # train minus val accuracy; healthy runs are ~0.01
OVERFIT_VAL_LOSS_RISE = 0.1  # val loss climbing this far above its minimum


def diagnose(run_id: str) -> Diagnosis:
    cfg = run_config(run_id)
    s = curve_summary(run_id)
    overlap = split_overlap(run_id)

    def found(bug: BugType, *evidence: str) -> Diagnosis:
        return Diagnosis(bug=bug, confidence=0.9, evidence=list(evidence))

    # 1. Leakage: direct evidence, so check it first.
    if overlap["n_overlap"] > 0:
        return found(BugType.LEAKAGE, f"{overlap['n_overlap']} examples in both train and val")

    # 2. Unnormalized inputs: visible in the config. Checked before the loss-explosion rule
    #    because raw 0-255 inputs also blow up the early loss, which would look like a bad LR.
    if cfg.get("normalize") is False:
        return found(BugType.UNNORMALIZED, "normalize=False in config")

    # 3. Learning rate too high: an explosion or an extreme setting.
    first_loss = s["train_loss_first_epoch"] or 0.0
    if s["training_status"] == "diverged" or first_loss > EXPLODED_FIRST_LOSS:
        return found(BugType.LR_TOO_HIGH, f"first-epoch train loss {first_loss}")
    if cfg.get("lr", 0) >= HIGH_LR:
        return found(BugType.LR_TOO_HIGH, f"learning rate {cfg['lr']}")

    # 4. Label shuffle: loss stuck at chance with a reasonable config.
    final_loss, val_acc = s["train_loss_final"], s["val_acc_final"]
    if (
        final_loss is not None
        and val_acc is not None
        and abs(final_loss - CHANCE_LOSS) < NEAR_CHANCE_LOSS
        and val_acc < NEAR_CHANCE_ACC
    ):
        return found(
            BugType.LABEL_SHUFFLE, f"final train loss {final_loss} near chance {CHANCE_LOSS}"
        )

    # 5. Overfitting: train far ahead of val, or val loss climbing.
    gap, rise = s["train_val_acc_gap"] or 0.0, s["val_loss_rise_since_min"] or 0.0
    if gap > OVERFIT_ACC_GAP or rise > OVERFIT_VAL_LOSS_RISE:
        return found(BugType.OVERFIT, f"train/val accuracy gap {gap}", f"val loss rise {rise}")

    return Diagnosis(bug=BugType.NONE, confidence=0.6, evidence=["no rule fired"])