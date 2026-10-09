"""Data structures passed between the investigator, diagnoser, and the rest of the graph.

Everything an LLM produces is one of these Pydantic models, so outputs are validated
and the eval can compare `Diagnosis.bug` to the ground truth with a simple equality check.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Reuse the training script's enum so bug names can never drift out of sync.
from ml_debug_agent.training.train import Bug as BugType

__all__ = [
    "BugType",
    "Diagnosis",
    "EvidenceReport",
    "DiagnoserDecision",
    "PipelineFix",
    "ConfigPatch",
    "RunReport",
    "PATCHABLE_FIELDS",
]

# Config fields the agent is allowed to change. The hidden data-pipeline fields are
# deliberately excluded: those are fixed through `PipelineFix` actions instead.
PATCHABLE_FIELDS = {
    "lr",
    "batch_size",
    "epochs",
    "hidden1",
    "hidden2",
    "dropout",
    "weight_decay",
    "train_subset",
    "normalize",
}


class Diagnosis(BaseModel):
    """The final answer about what is wrong with a training run.

    Field order matters: an LLM writes the JSON fields in this order, so `evidence`
    comes first. The model notes its observations before committing to a bug, instead
    of picking a label first and justifying it afterwards.
    """

    evidence: list[str] = Field(
        min_length=1,
        description="First, list observations from the evidence that look normal or "
        "abnormal, with numbers, e.g. 'train/val accuracy gap of 0.41 at the final epoch'.",
    )
    bug: BugType = Field(
        description="Then, the single most likely problem. Use 'none' if the run looks healthy."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are, from 0 to 1.")


class EvidenceReport(BaseModel):
    """What the investigator found. Raw tool outputs are kept so nothing is lost in the handoff."""

    config: dict[str, Any] | None = Field(default=None, description="Output of get_config.")
    curve_summary: dict[str, Any] | None = Field(
        default=None, description="Output of summarize_curves."
    )
    split_overlap: dict[str, Any] | None = Field(
        default=None, description="Output of check_split_overlap."
    )
    extra_findings: list[str] = Field(
        default_factory=list,
        description="Answers to the diagnoser's follow-up requests.",
    )
    investigator_notes: str = Field(
        default="", description="Which tools were used and anything notable."
    )


class DiagnoserDecision(BaseModel):
    """The diagnoser either commits to a diagnosis or asks the investigator for more evidence."""

    diagnosis: Diagnosis | None = Field(
        default=None, description="Set this when the evidence is sufficient."
    )
    follow_up_request: str | None = Field(
        default=None,
        description="Set this instead when more evidence is needed. Say exactly what to check.",
    )

    @model_validator(mode="after")
    def exactly_one(self) -> DiagnoserDecision:
        if (self.diagnosis is None) == (self.follow_up_request is None):
            raise ValueError("Set exactly one of 'diagnosis' or 'follow_up_request'.")
        return self


class PipelineFix(StrEnum):
    """Fixes for data-pipeline bugs, which can't be expressed as hyperparameter changes."""

    FIX_LABELS = "fix_labels"
    REMOVE_SPLIT_OVERLAP = "remove_split_overlap"


class ConfigPatch(BaseModel):
    """A proposed fix: hyperparameter overrides and/or a data-pipeline fix."""

    overrides: dict[str, float | int | bool | None] = Field(
        default_factory=dict,
        description=f"Config values to change. Allowed keys: {sorted(PATCHABLE_FIELDS)}.",
    )
    pipeline_fix: PipelineFix | None = Field(
        default=None, description="A data-pipeline fix, if the problem is in the data."
    )
    rationale: str = Field(description="Why this change should fix the diagnosed problem.")

    @field_validator("overrides")
    @classmethod
    def only_patchable(cls, v: dict[str, Any]) -> dict[str, Any]:
        unknown = set(v) - PATCHABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot patch {sorted(unknown)}. Allowed: {sorted(PATCHABLE_FIELDS)}")
        return v


class RunReport(BaseModel):
    """The final output of the full loop for one run."""

    source_run_id: str
    diagnosis: Diagnosis
    patch: ConfigPatch | None = None
    rerun_id: str | None = None
    before: dict[str, float] = Field(default_factory=dict)
    after: dict[str, float] | None = None
    improved: bool | None = None
    summary: str = ""