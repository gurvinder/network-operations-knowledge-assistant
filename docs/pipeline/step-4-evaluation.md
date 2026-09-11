# Step 4: Evaluation

**Layer:** Fine-Tuning Layer
**Components:** [Eval Hub](https://github.com/eval-hub/eval-hub)

## Overview

Eval Hub evaluates the fine-tuned model across three configurations — base model, OSFT-tuned, and OSFT-tuned + RAG — to measure the impact of each stage. Models that pass evaluation are promoted to the Model Registry.

## Prerequisites

- Eval Hub installed
- Checkpoint from Step 3 in S3
- vLLM serving the base model and fine-tuned model
- RAG pipeline running (for OSFT+RAG evaluation)

## Procedure

### 4.1 Define Evaluation Configurations

```python
configurations = [
    {
        "name": "base",
        "model_endpoint": "http://vllm-base:8000/v1",
        "description": "GPT OSS 20B without fine-tuning",
    },
    {
        "name": "osft-tuned",
        "model_endpoint": "http://vllm-tuned:8000/v1",
        "description": "GPT OSS 20B + OSFT knowledge tuning",
    },
    {
        "name": "osft-tuned+rag",
        "model_endpoint": "http://ogx-rag:8000/v1",
        "description": "GPT OSS 20B + OSFT + RAG retrieval",
    },
]
```

### 4.2 Run Evaluation

```python
from eval_hub import EvalHub

hub = EvalHub()

results = hub.evaluate(
    configurations=configurations,
    eval_dataset="s3://noc-pipeline/eval/noc-questions.jsonl",
    metrics=["accuracy", "completeness", "relevance"],
)
```

### 4.3 Review Results

```python
for config_name, scores in results["scores"].items():
    avg = sum(scores.values()) / len(scores)
    print(f"{config_name}: {scores} (avg: {avg:.2f})")
```

Expected output pattern:

| Configuration | Accuracy | Completeness | Relevance | Average |
|--------------|----------|-------------|-----------|---------|
| base | 0.45 | 0.40 | 0.50 | 0.45 |
| osft-tuned | 0.78 | 0.72 | 0.80 | 0.77 |
| osft-tuned+rag | 0.92 | 0.88 | 0.94 | 0.91 |

### 4.4 Gate Check

```python
PASS_THRESHOLD = 0.70

best_config = results["best_config"]
best_avg = sum(results["scores"][best_config].values()) / 3

if best_avg >= PASS_THRESHOLD:
    print(f"PASS: {best_config} scored {best_avg:.2f}, promoting to registry")
else:
    print(f"FAIL: best score {best_avg:.2f} below threshold {PASS_THRESHOLD}")
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| Evaluation report (JSON) | S3 (`s3://noc-pipeline/eval-results/`) | Step 5 (decision gate) |
| Pass/fail signal | Pipeline orchestration | Step 5 (registry promotion) |

## Next Step

[Step 5: Model Registry and Serving →](step-5-model-serving.md)
