"""Single-agent diagnoser: one LLM investigates a run with tools, then commits to a Diagnosis.

Two phases:
  1. Investigate: a tool-calling agent (LangChain's create_agent) gathers evidence.
  2. Answer: the same model reads the collected evidence and returns a structured Diagnosis.

Try it on one benchmark run:
    uv run python -m ml_debug_agent.agents.single_agent            # first benchmark run
    uv run python -m ml_debug_agent.agents.single_agent <run_id>
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ml_debug_agent.agents.llm import get_llm
from ml_debug_agent.agents.mlflow_access import list_benchmark_runs
from ml_debug_agent.agents.schemas import Diagnosis
from ml_debug_agent.agents.tools import build_tools

# What each label means, so the model knows the answer space. Deliberately definitions
# only, with no thresholds or symptoms: recognizing the signs is the agent's job.
LABEL_DEFINITIONS = """\
- none: the run is healthy; nothing is wrong.
- lr_too_high: the learning rate is too large for stable training.
- overfit: the model memorizes the training data and fails to generalize.
- label_shuffle: the training labels are wrong or don't match the inputs.
- leakage: validation data leaked into the training set.
- unnormalized: the input features were not normalized before training."""

INVESTIGATE_PROMPT = """\
You are an ML engineer debugging a training run. The run trained a small neural network \
(a multi-layer perceptron) on Fashion-MNIST, a 10-class image classification dataset. \
The run may have a problem, or it may be healthy.

Before concluding, you must call all three of these tools: summarize_curves, get_config, \
and check_split_overlap. You may also call get_metric_history to look at a curve in more \
detail. Then write a short summary of what you found, including the specific numbers \
that matter."""

ANSWER_PROMPT = f"""\
You are an ML engineer diagnosing a training run from the evidence below.

Choose the single most likely problem from these labels:
{LABEL_DEFINITIONS}

Choose "none" if the run looks healthy.

Respond with JSON with these fields, in this order:
- evidence: a list of observations, each citing specific numbers, noting what looks
  normal and what looks abnormal. Write these BEFORE choosing a label.
- bug: the single most likely label, based on your observations.
- confidence: how sure you are, from 0 to 1."""


def investigate(run_id: str, llm: BaseChatModel | None = None) -> list:
    """Phase 1: let the agent call tools until it decides it has enough evidence."""
    agent = create_agent(
        model=llm or get_llm(),
        tools=build_tools(run_id),
        system_prompt=INVESTIGATE_PROMPT,
    )
    result = agent.invoke(
        {"messages": [HumanMessage("Investigate this training run.")]},
        config={"recursion_limit": 20},  # cap on steps, in case the model loops
    )
    return result["messages"]


def format_evidence(messages: list, include_summary: bool = False) -> str:
    """Turn the investigation transcript into a compact evidence document.

    By default only the raw tool outputs are passed on. The investigator's own summary
    is left out because it can contain wrong interpretations (e.g. calling a full
    train/val overlap "expected") that the answer step then anchors on.
    """
    sections = [f"### {m.name}\n{m.content}" for m in messages if isinstance(m, ToolMessage)]
    final = messages[-1]
    if include_summary and isinstance(final, AIMessage) and final.content and not final.tool_calls:
        sections.append(f"### Investigator's summary\n{final.content}")
    return "\n\n".join(sections) or "No evidence was gathered."


def answer(evidence: str, llm: BaseChatModel | None = None) -> Diagnosis:
    """Phase 2: read the evidence and commit to one structured Diagnosis."""
    structured = (llm or get_llm()).with_structured_output(Diagnosis)
    return structured.invoke(
        [SystemMessage(ANSWER_PROMPT), HumanMessage(f"Evidence:\n\n{evidence}")]
    )


def diagnose(run_id: str) -> Diagnosis:
    """Entry point for the eval harness, with thinking mode off."""
    return answer(format_evidence(investigate(run_id)))


def diagnose_thinking(run_id: str) -> Diagnosis:
    """The same agent with Gemma 4's thinking mode on, for comparison."""
    llm = get_llm(thinking=True)
    return answer(format_evidence(investigate(run_id, llm)), get_llm())


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else list_benchmark_runs()[0]
    messages = investigate(target)
    print("Tools called:", [m.name for m in messages if isinstance(m, ToolMessage)])
    evidence = format_evidence(messages)
    print(f"\n--- Evidence passed to the answer step ---\n{evidence}\n")
    print(answer(evidence).model_dump_json(indent=2))