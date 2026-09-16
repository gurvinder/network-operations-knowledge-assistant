# Step 2: Synthetic Data Generation

**Layer:** Fine-Tuning Layer
**Components:** [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub), [vLLM](https://github.com/vllm-project/vllm) (Teacher Model)

## Overview

SDG Hub consumes the structured datasets produced by the Data Layer and generates synthetic Q&A training pairs using a teacher model. The output is quality-filtered JSONL in OpenAI messages format, ready for fine-tuning. Three of the four knowledge tuning flows include faithfulness filtering as a final block — the Key Facts flow does not.

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

### 2.2 Discover Available Knowledge Tuning Flows

```python
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()
flows = FlowRegistry.list_flows()
```

SDG Hub provides 4 English knowledge tuning flows. This pipeline uses 2 of the 4 (the remaining 2 may be used for more specialized generation if needed):

| Flow | ID | Faithfulness Filter | Used Here |
|------|-----|-------------------|-----------|
| Key Facts | `heavy-heart-77` | No | Yes (bootstrap) |
| Extractive Summary Knowledge Tuning | `epic-jade-656` | Yes | Yes (full-scale) |
| Detailed Summary Knowledge Tuning | `mild-thunder-748` | Yes | No |
| Document Based Knowledge Tuning | `stellar-peak-605` | Yes | No |

### 2.3 Load and Configure Key Facts Flow (Bootstrap)

```python
from sdg_hub import FlowRegistry, Flow

FlowRegistry.discover_flows()

flow_path = FlowRegistry.get_flow_path_safe("heavy-heart-77")
flow = Flow.from_yaml(flow_path)

flow.set_model_config(
    model="hosted_vllm/gpt-oss-120b",
    api_base="http://vllm-teacher:8000/v1",
    api_key="your-key",
)

bootstrap_output = flow.generate(ds, max_concurrency=50)
```

> **Note:** The Key Facts flow does not include faithfulness filtering. If you need filtered output from this flow, apply post-processing manually.

### 2.4 Load and Run Full Knowledge Tuning Flow

```python
flow_path = FlowRegistry.get_flow_path_safe("epic-jade-656")
flow = Flow.from_yaml(flow_path)

flow.set_model_config(
    model="hosted_vllm/gpt-oss-120b",
    api_base="http://vllm-teacher:8000/v1",
    api_key="your-key",
)

full_output = flow.generate(ds, max_concurrency=50)
```

> **Note:** The ~2,000 sample target is a configurable choice. The number of generated samples per document is controlled by the `n` parameter on LLM blocks, overridable via `runtime_params`:
> ```python
> full_output = flow.generate(
>     ds,
>     runtime_params={"gen_extractive_summary": {"n": 50}},
>     max_concurrency=50,
> )
> ```

### 2.5 Export as Messages JSONL

The Extractive Summary flow produces quality-filtered output (faithfulness evaluated as YES/NO, filtering applied as the final block). The output is ready for training.

```python
import json

with open("sdg_output.jsonl", "w") as f:
    for row in full_output.to_dict(orient="records"):
        record = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["response"]},
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
