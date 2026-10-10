"""Two-agent diagnoser: an investigator gathers evidence, a diagnoser interprets it.

    START → investigator → diagnoser ──(diagnosis)────────────────→ END
                 ▲             │
                 └──(follow-up request, at most MAX_FOLLOW_UPS)─┘

- Investigator (tools, never interprets): round 1 calls the three required tools. Later
  rounds answer the diagnoser's specific request. Tool outputs are copied into the
  EvidenceReport by code, never retyped by the model, and the investigator's own prose is
  discarded so it can't plant an interpretation.
- Diagnoser (no tools): reasons in free text, ending with either "Final answer: <label>"
  or "Need more evidence: <request>". Code parses that line into a DiagnoserDecision.
  Ollama doesn't show the model its JSON schema, so the label comes from the free-text
  line, and structured output is only used to pull out the evidence and confidence.

Try it on one benchmark run (prints the full trace):
    uv run python -m ml_debug_agent.agents.graph            # first benchmark run
    uv run python -m ml_debug_agent.agents.graph <run_id>
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ml_debug_agent.agents.llm import get_llm
from ml_debug_agent.agents.mlflow_access import list_benchmark_runs
from ml_debug_agent.agents.schemas import BugType, DiagnoserDecision, Diagnosis, EvidenceReport
from ml_debug_agent.agents.single_agent import LABEL_DEFINITIONS
from ml_debug_agent.agents.tools import build_tools

MAX_FOLLOW_UPS = 2
REQUIRED_TOOLS = {  # tool name -> EvidenceReport field it fills
    "summarize_curves": "curve_summary",
    "get_config": "config",
    "check_split_overlap": "split_overlap",
}

INVESTIGATOR_PROMPT = """\
You gather evidence about a training run for someone else to diagnose. The run trained a \
small neural network (a multi-layer perceptron) on Fashion-MNIST, a 10-class image \
classification dataset.

Call all three of these tools: summarize_curves, get_config, and check_split_overlap. \
Do not interpret the results or guess what is wrong; just collect them."""

FOLLOW_UP_PROMPT = """\
You gather evidence about a training run for someone else to diagnose. They have asked \
for more detail. Use the tools to answer their request as directly as possible. \
Do not interpret the results or guess what is wrong; just collect them."""

DIAGNOSER_PROMPT = f"""\
You are an ML engineer diagnosing a training run. The run trained a small neural network \
(a multi-layer perceptron) on Fashion-MNIST, a 10-class image classification dataset. \
A colleague collected the evidence below; you cannot run tools yourself.

The possible labels are:
{LABEL_DEFINITIONS}

Think it through in plain text first. Go through the evidence and say what each important \
number tells you about whether training worked. Then weigh each label against it.

End with exactly one of these lines:
Final answer: <label>
Need more evidence: <one specific thing to check, e.g. a metric's per-epoch values>"""

MUST_COMMIT = """

You have no more follow-up requests left, so you must end with "Final answer: <label>"."""

EXTRACT_PROMPT = """\
Below is an ML engineer's reasoning about a training run. Extract their conclusion: the \
observations they relied on (each with its numbers), the label they chose, and how \
confident they sounded, from 0 to 1."""


class GraphState(TypedDict, total=False):
    run_id: str
    evidence: EvidenceReport
    pending_request: str | None
    rounds: int  # follow-up rounds completed
    diagnosis: Diagnosis | None
    diagnoser_reasoning: list[str]  # one entry per diagnoser turn, for failure analysis


# --- Investigator -------------------------------------------------------------------------


def _tool_results(messages: list) -> list[tuple[str, dict[str, Any], str]]:
    """(tool name, arguments, output) for every tool call in an agent transcript."""
    args_by_id = {
        call["id"]: call["args"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    }
    return [
        (m.name, args_by_id.get(m.tool_call_id, {}), m.content)
        for m in messages
        if isinstance(m, ToolMessage)
    ]


def _run_tool_agent(llm: BaseChatModel, run_id: str, system: str, request: str) -> list:
    agent = create_agent(model=llm, tools=build_tools(run_id), system_prompt=system)
    result = agent.invoke(
        {"messages": [HumanMessage(request)]},
        config={"recursion_limit": 20},  # cap on steps, in case the model loops
    )
    return result["messages"]


def first_investigation(run_id: str, llm: BaseChatModel) -> EvidenceReport:
    """Round 1: the investigator calls the required tools; code files the outputs."""
    messages = _run_tool_agent(llm, run_id, INVESTIGATOR_PROMPT, "Collect the evidence.")
    report = EvidenceReport()
    called = []
    for name, _, output in _tool_results(messages):
        called.append(name)
        if name in REQUIRED_TOOLS:
            setattr(report, REQUIRED_TOOLS[name], json.loads(output))

    # A skipped required tool is filled in by code, so the diagnoser always gets the
    # basics. The note records it, so a lazy investigator still shows up in the analysis.
    tools = {t.name: t for t in build_tools(run_id)}
    missed = [name for name in REQUIRED_TOOLS if name not in called]
    for name in missed:
        setattr(report, REQUIRED_TOOLS[name], json.loads(tools[name].invoke({})))
    report.investigator_notes = f"Round 1 tools called: {called or 'none'}." + (
        f" Filled in by code: {missed}." if missed else ""
    )
    return report


def follow_up(run_id: str, request: str, llm: BaseChatModel) -> list[str]:
    """A later round: the investigator answers one request. Returns the raw tool outputs."""
    messages = _run_tool_agent(llm, run_id, FOLLOW_UP_PROMPT, f"Request: {request}")
    findings = [
        f"{name}({json.dumps(args)}) -> {output}" for name, args, output in _tool_results(messages)
    ]
    return findings or [f"No tool could answer the request: {request}"]


# --- Diagnoser ----------------------------------------------------------------------------


def format_report(report: EvidenceReport) -> str:
    sections = [
        f"### summarize_curves\n{json.dumps(report.curve_summary)}",
        f"### get_config\n{json.dumps(report.config)}",
        f"### check_split_overlap\n{json.dumps(report.split_overlap)}",
    ]
    if report.extra_findings:
        sections.append("### Follow-up findings\n" + "\n".join(report.extra_findings))
    return "\n\n".join(sections)


_FINAL = re.compile(r"final answer\W*([a-z_]+)", re.IGNORECASE)
_MORE = re.compile(r"need more evidence\W*(.+)", re.IGNORECASE)


def parse_decision(text: str) -> tuple[BugType | None, str | None]:
    """Read the reasoning's closing line: (label, None), (None, request), or (None, None)."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("*`_ ")  # small models like to bold the answer
        if m := _FINAL.search(line):
            label = m.group(1).lower()
            return (BugType(label), None) if label in BugType else (None, None)
        if m := _MORE.search(line):
            return None, m.group(1).strip()
    return None, None


def decide(
    report: EvidenceReport, rounds: int, llm: BaseChatModel
) -> tuple[DiagnoserDecision, str]:
    """One diagnoser turn: free-text reasoning, then a validated decision."""
    can_ask = rounds < MAX_FOLLOW_UPS
    system = DIAGNOSER_PROMPT + ("" if can_ask else MUST_COMMIT)
    reasoning = llm.invoke(
        [SystemMessage(system), HumanMessage(f"Evidence:\n\n{format_report(report)}")]
    ).text
    label, request = parse_decision(reasoning)

    if request and can_ask:
        return DiagnoserDecision(follow_up_request=request), reasoning

    extracted = llm.with_structured_output(Diagnosis).invoke(
        [SystemMessage(EXTRACT_PROMPT), HumanMessage(reasoning)]
    )
    if label is not None:  # the free-text line is the decision; extraction only fills details
        extracted = extracted.model_copy(update={"bug": label})
    return DiagnoserDecision(diagnosis=extracted), reasoning


# --- Graph --------------------------------------------------------------------------------


def build_graph(investigator_llm: BaseChatModel, diagnoser_llm: BaseChatModel):
    def investigator(state: GraphState) -> GraphState:
        if state.get("evidence") is None:
            return {"evidence": first_investigation(state["run_id"], investigator_llm)}
        findings = follow_up(state["run_id"], state["pending_request"], investigator_llm)
        report = state["evidence"].model_copy(
            update={"extra_findings": [*state["evidence"].extra_findings, *findings]}
        )
        return {"evidence": report, "pending_request": None, "rounds": state["rounds"] + 1}

    def diagnoser(state: GraphState) -> GraphState:
        decision, reasoning = decide(state["evidence"], state["rounds"], diagnoser_llm)
        return {
            "diagnosis": decision.diagnosis,
            "pending_request": decision.follow_up_request,
            "diagnoser_reasoning": [*state.get("diagnoser_reasoning", []), reasoning],
        }

    def route(state: GraphState) -> str:
        return "investigator" if state.get("diagnosis") is None else END

    graph = StateGraph(GraphState)
    graph.add_node("investigator", investigator)
    graph.add_node("diagnoser", diagnoser)
    graph.add_edge(START, "investigator")
    graph.add_edge("investigator", "diagnoser")
    graph.add_conditional_edges("diagnoser", route, ["investigator", END])
    return graph.compile()


def run_graph(
    run_id: str,
    investigator_llm: BaseChatModel | None = None,
    diagnoser_llm: BaseChatModel | None = None,
) -> GraphState:
    """Run the full loop and return the final state, including the reasoning trace."""
    graph = build_graph(investigator_llm or get_llm(), diagnoser_llm or get_llm())
    return graph.invoke({"run_id": run_id, "evidence": None, "rounds": 0})


def diagnose(run_id: str) -> Diagnosis:
    """Entry point for the eval harness."""
    return run_graph(run_id)["diagnosis"]


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else list_benchmark_runs()[0]
    state = run_graph(target)
    print(state["evidence"].investigator_notes)
    for i, reasoning in enumerate(state["diagnoser_reasoning"], 1):
        print(f"\n--- Diagnoser turn {i} ---\n{reasoning}")
    if state["evidence"].extra_findings:
        print("\n--- Follow-up findings ---")
        print("\n".join(state["evidence"].extra_findings))
    print(f"\nFollow-up rounds: {state['rounds']}")
    print(state["diagnosis"].model_dump_json(indent=2))
