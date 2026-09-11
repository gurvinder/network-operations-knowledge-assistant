# Step 2: Synthetic Data Generation

**Layer:** Fine-Tuning Layer
**Components:** [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub), [vLLM](https://github.com/vllm-project/vllm) (Teacher Model)

## Overview

SDG Hub consumes the structured datasets produced by the Data Layer and generates synthetic Q&A training pairs using a teacher model. The output is quality-filtered JSONL in OpenAI messages format, ready for fine-tuning.

## Prerequisites

- SDG Hub installed: `pip install sdg-hub`
- vLLM serving gpt-oss-120b (teacher model) on an OpenAI-compatible endpoint
- Structured datasets from Step 1 in S3

## Procedure

### 2.1 Load Structured Datasets

```python
from datasets import Dataset

ds = Dataset.load_from_disk("s3://noc-pipeline/datasets/structured/")
# Columns: document, document_outline, domain
```

### 2.2 Configure SDG Hub

```python
from sdg_hub import SDGHub

hub = SDGHub()

# List available knowledge tuning flows
flows = hub.list_flows()
```

SDG Hub provides two flows for knowledge tuning:
- **Key Facts** — extracts factual Q&A pairs for initial bootstrap (smaller, higher quality)
- **Extractive Summary Knowledge Tuning** — full-scale generation with varied question types

### 2.3 Run Key Facts Flow (Bootstrap)

```python
flow = hub.load_flow("key-facts")

bootstrap_output = flow.run(
    dataset=ds,
    teacher_model_endpoint="http://vllm-teacher:8000/v1",
    num_samples=200,
)
```

### 2.4 Run Full Knowledge Tuning Flow

```python
flow = hub.load_flow("extractive-summary-knowledge-tuning")

full_output = flow.run(
    dataset=ds,
    teacher_model_endpoint="http://vllm-teacher:8000/v1",
    num_samples=2000,
)
```

### 2.5 Quality Filtering

SDG Hub applies faithfulness and relevancy scoring automatically. Filter outputs:

```python
filtered = full_output.filter(
    lambda row: row.get("faithfulness_judgment", True)
    and row.get("relevancy_score", 1.0) >= 0.7
)
```

### 2.6 Export as Messages JSONL

```python
import json

with open("sdg_output.jsonl", "w") as f:
    for row in filtered:
        record = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }
        f.write(json.dumps(record) + "\n")
```

## Output Format

Each line in the JSONL file:

```json
{"messages": [{"role": "user", "content": "What should a NOC engineer do when..."}, {"role": "assistant", "content": "The engineer should first verify..."}]}
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| `sdg_output.jsonl` (messages format) | S3 (`s3://noc-pipeline/sdg-output/`) | Step 3 (Model Training) |

## Next Step

[Step 3: Model Training →](step-3-model-training.md)
