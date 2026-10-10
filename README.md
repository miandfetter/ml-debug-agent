# ml-debug-agent

LLM agents that diagnose what went wrong in a machine learning training run, using only
what an engineer would see in the experiment tracker: the logged config, per-epoch
metrics, and the train/validation split.

The project builds a labeled benchmark of deliberately broken training runs, then
compares three diagnosers on it: a hand-written rule baseline, a single tool-using agent,
and a two-agent LangGraph system in which a diagnoser can ask an investigator for more
evidence.

**Stack:** PyTorch · MLflow · LangChain · LangGraph · Pydantic · Ollama / Gemini / Amazon Bedrock ·
Docker · AWS (S3, ECR) · pytest · uv

## How it works

```
 train.py ──► MLflow ──► mlflow_access.py ──► tools.py ──► agent ──► Diagnosis ──► eval harness
 (inject a    (runs,     (the only code that  (measurements,  (single or       (validated   (accuracy,
  known bug)   metrics)   reads MLflow; hides  no conclusions)  two-agent)       Pydantic     confusion
                          the answer)                                            model)       matrix)
```

### 1. A benchmark with known answers

A small MLP is trained on Fashion-MNIST (10 classes). Each bug is a set of config changes
from a healthy run:

| Bug | What is changed |
|---|---|
| `none` | nothing (healthy control) |
| `lr_too_high` | learning rate 0.5, about 500x normal |
| `overfit` | 500 training examples, no dropout or weight decay, 30 epochs |
| `label_shuffle` | training labels randomly permuted |
| `leakage` | the validation set copied into the training set |
| `unnormalized` | raw 0-255 pixels instead of normalized inputs |

`benchmark-v1` contains **30 runs: 6 bug types x 5 seeds**, with small variations in the
healthy settings so runs of the same bug don't look identical. Every run logs its config,
per-epoch losses, accuracies and gradient norms, and its split indices to MLflow.

### 2. Keeping the answer hidden

The true bug is stored in each run's name, description and an `eval.true_bug` tag.
[`mlflow_access.py`](src/ml_debug_agent/agents/mlflow_access.py) is the only module that
reads MLflow. Agents see runs through `get_run_view()`, which uses an **allowlist** of
fields, so the parameters that *are* the bug (`label_noise`, `leak_val_into_train`) are
never exposed. Only the eval harness may call `get_true_bug()`.

### 3. Tools

[`tools.py`](src/ml_debug_agent/agents/tools.py) gives the agents four tools:
`summarize_curves`, `get_config`, `check_split_overlap` and `get_metric_history`.
Code does the arithmetic, since LLMs are unreliable at maths over long lists of numbers,
and the run ID is fixed in code, so the model can't mistype it or inspect other runs.
Tools return **measurements, never conclusions**.

### 4. A fair comparison

Prompts and tool descriptions define what each label means, but never describe symptoms
or thresholds. Otherwise the agent would just be the baseline's rules written in English.
A test scans every prompt and tool description for give-away terms.

### 5. The diagnosers

- **Rule baseline** ([`baseline.py`](src/ml_debug_agent/eval/baseline.py)): hand-written
  if-statements over the same tool outputs.
- **Single agent** ([`single_agent.py`](src/ml_debug_agent/agents/single_agent.py)): a
  tool-calling agent gathers evidence, then a second call reads only the raw tool outputs
  and returns a structured `Diagnosis`.
- **Two agents** ([`graph.py`](src/ml_debug_agent/agents/graph.py)): a LangGraph state
  machine.

```mermaid
graph TD;
    START([start]) --> investigator;
    investigator --> diagnoser;
    diagnoser -.->|follow-up request, max 2| investigator;
    diagnoser -.->|diagnosis| END([end]);
```

The **investigator** has tools but never interprets. Code copies its tool outputs into an
`EvidenceReport`, and fills in any required tool it skipped. The **diagnoser** has no
tools. It reasons in plain text and ends with either `Final answer: <label>` or
`Need more evidence: <request>`. On its last allowed turn it is told it must commit.
Keeping the two roles separate makes each failure traceable: was the evidence missing,
or was it misread?

### 6. Evaluation

[`run_benchmark.py`](src/ml_debug_agent/eval/run_benchmark.py) runs any diagnoser over
the benchmark and reports accuracy, accuracy per bug, false-positive rate (healthy runs
flagged as buggy) and a confusion matrix. A failed run is recorded as `error` rather than
stopping the eval.

### 7. Running in the cloud (in progress)

- **Artifacts in S3.** MLflow keeps run metadata (params, metrics, tags) in a local SQLite
  database and run files in an artifact store. Setting `MLFLOW_ARTIFACT_ROOT` makes new
  experiments store their files in a private, encrypted S3 bucket. `benchmark-v2` is the
  same 30 runs as v1 with its artifacts in S3; the agents' tools read them through MLflow
  with no code changes.
- **A slim evaluation image.** The [Dockerfile](Dockerfile) builds an image that runs the
  eval harness. Bug and config definitions live in a PyTorch-free module
  ([`config.py`](src/ml_debug_agent/training/config.py)) and PyTorch is an optional
  dependency group, so the image doesn't include it. A multi-stage build ships only the
  finished Python environment.
- **ECR.** The image is stored in a private Amazon ECR repository with vulnerability
  scanning on push and a rule that keeps only the 5 newest images.
- **Amazon Bedrock** is a third model provider (Amazon Nova 2 Lite by default). It's
  implemented and unit-tested; end-to-end runs are waiting on a Bedrock quota for the new
  AWS account.
- **Credentials:** AWS IAM Identity Center (SSO) locally, never long-lived access keys.

## Results so far

All results are on `benchmark-v1` unless noted.

| Diagnoser | Model | Runs | Correct |
|---|---|---|---|
| Rule baseline | none | 30 | **30 / 30** |
| Rule baseline, in Docker | none | 30 (`benchmark-v2`, artifacts in S3) | **30 / 30** |
| Single agent | gemma4:e2b (2B, local) | 30 | 5 / 30 (predicted `none` every time) |
| Single agent | gemini-2.5-flash | 6 (1 per bug) | 5 / 5 completed; the 6th hit the free-tier quota |
| Single agent | qwen3.5:9b (local) | 30 | **20 / 30** |
| Two agents | qwen3.5:9b (local) | 6 (1 per bug) | **5 / 6** |

Single agent with qwen3.5:9b, by bug:

| none | lr_too_high | overfit | label_shuffle | leakage | unnormalized |
|---|---|---|---|---|---|
| 5/5 | 4/5 | 4/5 | **0/5** | 5/5 | 2/5 |

The prompts and tool descriptions changed between some of these runs, and the 6-run
subsets are small, so treat the comparisons as indicative rather than final.

### What I learned

- **The rule baseline scores 100% because its author designed the bugs.** Its thresholds
  are tuned to this benchmark. The agents get no thresholds, so they are tested on
  reasoning from evidence, which is what matters for bugs nobody wrote a rule for. A
  natural next step is adding bug types the baseline has no rule for.
- **Structured output on Ollama hid the schema from the model.** Ollama enforces a JSON
  schema while generating text but never shows it to the model, so the `Diagnosis` field
  descriptions (and the "write evidence before choosing a label" design) were silently
  ignored, and the 2B model answered `none` for everything. Describing the schema in the
  prompt fixed this.
- **`label_shuffle` is hard because its only symptom is an absence.** Training loss stays
  at ln(10) ≈ 2.30, the loss of random guessing over 10 classes, and accuracy at ~10%.
  Nothing diverges and there is no train/validation gap, so a model scanning for warning
  signs finds none, and the noisy validation loss looks like overfitting. The single agent
  got 0/5. In the two-agent system, the diagnoser's free-text reasoning compared the loss
  with chance level and got it right.
- **A hint was hiding in a tool description** ("a nonzero overlap means leakage"). It was
  removed, and the fairness test now covers tool descriptions as well as prompts.
- **The first Docker image was 3.66 GB.** uv's download cache had been baked into a layer,
  and a `chown -R` stored a second copy of every installed file. A build cache mount, no
  recursive `chown`, and a multi-stage build brought it to 1.41 GB (310 MB compressed in
  ECR). Most of the rest is full MLflow's extras.
- **Docker Desktop pushes an index plus a build attestation by default.** The extra
  untagged entries count against ECR's cleanup rule, and ECR's basic scanning doesn't scan
  an index. Building with `--provenance=false` pushes a single plain image.
- **A new AWS account had Bedrock quotas of 0 for every model.** Comparing the applied
  quotas with AWS's defaults (`list-aws-default-service-quotas`) showed the limits were
  account overrides rather than a code problem.

## Running it

### Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) for local models, or a Gemini API key
- About 7 GB of disk space for `qwen3.5:9b`; 16 GB of RAM is enough to run it
- For the cloud parts (optional): [Docker](https://www.docker.com/) and AWS CLI v2 with an
  SSO profile (`aws configure sso`)

### Setup

```bash
uv sync
ollama pull qwen3.5:9b
```

`uv sync` installs everything, including PyTorch (the `train` group) and the dev tools.

Settings go in a `.env` file in the project root (it is gitignored). For example:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
BENCHMARK_EXPERIMENT=benchmark-v2
AWS_PROFILE=ml-debug
MLFLOW_ARTIFACT_ROOT=s3://<your-bucket>/mlflow
```

| Setting | What it does | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama`, `gemini` or `bedrock` | `ollama` |
| `OLLAMA_MODEL` / `GEMINI_MODEL` / `BEDROCK_MODEL` | Model for that provider | `gemma4:e2b` / `gemini-2.5-flash` / `us.amazon.nova-2-lite-v1:0` |
| `BEDROCK_REGION` | Region for Bedrock calls | `us-east-1` |
| `BENCHMARK_EXPERIMENT` | Benchmark that commands use when none is named | `benchmark-v1` |
| `AWS_PROFILE` | AWS CLI profile for S3 and Bedrock | none |
| `MLFLOW_ARTIFACT_ROOT` | Where *new* experiments store run files, e.g. an S3 path | MLflow's local `mlruns/` |

Gemini also needs `GEMINI_API_KEY` in your environment. AWS access uses your SSO login
(`aws sso login`), never access keys.

### Generate the benchmark

Downloads Fashion-MNIST to `data/` and trains 30 runs into an MLflow experiment. It is
safe to re-run: existing runs are skipped.

```bash
uv run python scripts/generate_benchmark.py --experiment benchmark-v2
```

If `MLFLOW_ARTIFACT_ROOT` is set when the experiment is first created, its artifacts go to
`<root>/<experiment>` (for example in S3). MLflow fixes this when the experiment is
created, so set it beforehand. To browse the runs at http://localhost:5000:

```bash
uv run mlflow ui
```

To train a single run with a specific bug:

```bash
uv run python -m ml_debug_agent.training.train --bug overfit --seed 1
```

### Evaluate a diagnoser

The full benchmark:

```bash
uv run python -m ml_debug_agent.eval.run_benchmark --system baseline
uv run python -m ml_debug_agent.eval.run_benchmark --system single_agent
uv run python -m ml_debug_agent.eval.run_benchmark --system two_agent
```

Quicker subsets: one run of each bug type (about 20 minutes for `two_agent` on a laptop),
or only some bug types:

```bash
uv run python -m ml_debug_agent.eval.run_benchmark --system two_agent --per-bug 1
uv run python -m ml_debug_agent.eval.run_benchmark --system two_agent --bugs label_shuffle lr_too_high
```

Systems: `baseline`, `single_agent`, `single_agent_thinking`, `two_agent`. Use
`--experiment` to pick a benchmark other than the default. Per-run results (prediction,
evidence, confidence, time, errors) are saved to
`results/<experiment>/<system>[_per_bugN][_bugs].csv`.

### Inspect one diagnosis

The full two-agent trace (tools called, the diagnoser's reasoning on each turn, any
follow-up findings) for the benchmark's first run, or for a given run:

```bash
uv run python -m ml_debug_agent.agents.graph
uv run python -m ml_debug_agent.agents.graph <run_id>
uv run python -m ml_debug_agent.agents.single_agent <run_id>
```

Exactly what the agents are allowed to see for a run:

```bash
uv run python -m ml_debug_agent.agents.mlflow_access <run_id>
```

### Tests

```bash
uv run pytest
```

The tests use scripted fake models, so they need no LLM, and train tiny stand-in runs in a
throwaway MLflow store. They need no AWS access either: they never write to S3 and only
construct (never call) the Bedrock client. They cover the tools, the eval harness, the
agents' plumbing, the two-agent routing and follow-up cap, the model providers, the
artifact-location logic, and the no-give-away rule for prompts and tool descriptions.

### Docker

Build the evaluation image (it runs `run_benchmark`; PyTorch is not included):

```bash
docker build --provenance=false -t ml-debug-agent .
```

Run the rule baseline in the container against `benchmark-v2`. This mounts your
`mlflow.db` read-only, shares the `results/` folder, and passes short-lived credentials
from your SSO login as environment variables (nothing is written to disk or baked into the
image):

```bash
docker run --rm \
  -v "$PWD/mlflow.db:/app/mlflow.db:ro" \
  -v "$PWD/results:/app/results" \
  --env-file <(aws configure export-credentials --profile ml-debug --format env-no-export) \
  -e AWS_REGION=us-east-1 \
  -e BENCHMARK_EXPERIMENT=benchmark-v2 \
  ml-debug-agent --system baseline
```

Arguments after the image name replace the default `--system baseline`. The container
can't reach Ollama on your machine, so the agent systems need `-e LLM_PROVIDER=bedrock`
(or `gemini`).

### Push the image to ECR

One-time setup (already done for this project): a private repository with scanning on
push and a cleanup rule.

```bash
aws ecr create-repository --repository-name ml-debug-agent --image-scanning-configuration scanOnPush=true --encryption-configuration encryptionType=AES256
```

```bash
aws ecr put-lifecycle-policy --repository-name ml-debug-agent --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep only the 5 most recent images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}'
```

The push routine, after each code change. Replace `<account-id>` with your AWS account ID
(`aws sts get-caller-identity --query Account --output text`).

1. Build the image:

   ```bash
   docker build --provenance=false -t ml-debug-agent .
   ```

2. Log Docker in to ECR (the password comes from your SSO login and lasts 12 hours):

   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   ```

3. Tag the image with its ECR address:

   ```bash
   docker tag ml-debug-agent:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-debug-agent:latest
   ```

4. Push it (only changed layers are uploaded):

   ```bash
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-debug-agent:latest
   ```

Check what's in the repository, and the vulnerability scan results:

```bash
aws ecr describe-images --repository-name ml-debug-agent --query "imageDetails[].[imageTags[0],imageSizeInBytes,imageScanStatus.status]" --output table
```

```bash
aws ecr describe-image-scan-findings --repository-name ml-debug-agent --image-id imageTag=latest --query "imageScanFindings.findingSeverityCounts"
```

## Project layout

```
scripts/generate_benchmark.py   build the labeled benchmark
src/ml_debug_agent/
  training/config.py            bug types and training configs (no PyTorch)
  training/train.py             MLP training with injectable bugs, logged to MLflow
  agents/mlflow_access.py       the only MLflow reader; hides the answer
  agents/tools.py               LangChain tools over a run
  agents/schemas.py             Pydantic models for every LLM output
  agents/llm.py                 chooses Ollama, Gemini or Bedrock from .env
  agents/single_agent.py        single tool-using agent
  agents/graph.py               two-agent LangGraph system
  eval/baseline.py              rule-based baseline
  eval/run_benchmark.py         evaluation harness
tests/                          pytest suite
Dockerfile, .dockerignore       the evaluation image
```

## Next steps

- Run the evaluation as an ECS Fargate task that pulls the image from ECR, downloads
  `mlflow.db` from S3 at startup, and writes results back to S3
- Run the agents on Amazon Bedrock once the account's quota is available
- Define the AWS resources in Terraform, and add CI with GitHub Actions (tests on every
  push; build and push the image on merge, authenticating with OIDC rather than keys)
- Run the full 30-run benchmark for the two-agent system
- Add bug types the rule baseline has no rule for, to test generalization
- Shrink the image further by using `mlflow-skinny`

## License

MIT
