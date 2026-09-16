# Step 4: Evaluation

**Layer:** Fine-Tuning Layer
**Components:** [Eval Hub](https://github.com/eval-hub/eval-hub)

## Overview

Eval Hub evaluates the fine-tuned model across three configurations — base model, OSFT-tuned, and OSFT-tuned + RAG — to measure the impact of each stage. Models that pass evaluation are promoted to the Model Registry.

## Prerequisites

- Eval Hub SDK installed: `pip install "eval-hub-sdk[client]"`
- Checkpoint from Step 3 deployed on a vLLM endpoint
- Base model also deployed for comparison
- RAG pipeline running (for OSFT+RAG evaluation)

## Procedure

### 4.1 Configure the Eval Hub Client

```python
from evalhub import SyncEvalHubClient

client = SyncEvalHubClient(
    base_url="http://eval-hub.<your-namespace>.svc:8000",
)
```

### 4.2 Submit Evaluation Jobs

Submit a job for each model configuration:

```python
from evalhub.types import JobSubmissionRequest, ModelConfig, BenchmarkConfig

# Evaluate base model
base_job = client.jobs.submit(JobSubmissionRequest(
    name="noc-eval-base",
    model=ModelConfig(
        name="gpt-oss-20b-base",
        endpoint="http://vllm-base.<your-namespace>.svc/v1",
    ),
    benchmarks=[
        BenchmarkConfig(name="telecom-qa", num_samples=50),
    ],
))

# Evaluate OSFT-tuned model
tuned_job = client.jobs.submit(JobSubmissionRequest(
    name="noc-eval-osft-tuned",
    model=ModelConfig(
        name="gpt-oss-20b-noc-tuned",
        endpoint="http://gpt-oss-20b-noc-tuned.<your-namespace>.svc/v1",
    ),
    benchmarks=[
        BenchmarkConfig(name="telecom-qa", num_samples=50),
    ],
))

# Evaluate OSFT-tuned + RAG
rag_job = client.jobs.submit(JobSubmissionRequest(
    name="noc-eval-osft-rag",
    model=ModelConfig(
        name="gpt-oss-20b-noc-tuned-rag",
        endpoint="http://ogx-ai.<your-namespace>.svc:8321/v1",
    ),
    benchmarks=[
        BenchmarkConfig(name="telecom-qa", num_samples=50),
    ],
))
```

### 4.3 Wait for Results

```python
base_result = client.jobs.wait_for_completion(base_job.id)
tuned_result = client.jobs.wait_for_completion(tuned_job.id)
rag_result = client.jobs.wait_for_completion(rag_job.id)
```

### 4.4 Review Results

```python
for name, result in [("base", base_result), ("osft-tuned", tuned_result), ("osft-tuned+rag", rag_result)]:
    print(f"{name}: {result.scores}")
```

Expected output pattern:

| Configuration | Accuracy | Completeness | Relevance | Average |
|--------------|----------|-------------|-----------|---------|
| base | 0.45 | 0.40 | 0.50 | 0.45 |
| osft-tuned | 0.78 | 0.72 | 0.80 | 0.77 |
| osft-tuned+rag | 0.92 | 0.88 | 0.94 | 0.91 |

### 4.5 Gate Check

```python
PASS_THRESHOLD = 0.70

best_scores = rag_result.scores
best_avg = sum(best_scores.values()) / len(best_scores)

if best_avg >= PASS_THRESHOLD:
    print(f"PASS: osft-tuned+rag scored {best_avg:.2f}, promoting to registry")
else:
    print(f"FAIL: best score {best_avg:.2f} below threshold {PASS_THRESHOLD}")
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| Evaluation results (per job) | Eval Hub server | Step 5 (decision gate) |
| Pass/fail signal | Pipeline orchestration | Step 5 (registry promotion) |

## Next Step

[Step 5: Model Registry and Serving →](step-5-model-serving.md)
