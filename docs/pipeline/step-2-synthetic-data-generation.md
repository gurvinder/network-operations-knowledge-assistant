# Step 2: Synthetic Data Generation

**Layer:** Fine-Tuning Layer
**Components:** [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub), [vLLM](https://github.com/vllm-project/vllm) (Teacher Model)
**Reference:** [knowledge_generation.ipynb](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb)

## Overview

SDG Hub consumes the structured datasets produced by the Data Layer and generates synthetic Q&A training pairs using a teacher model across 4 knowledge tuning flows. Each flow produces a different type of document augmentation and Q&A pairs. The outputs are then combined via a mixing step into a final training dataset. Quality filtering (faithfulness YES/NO) is built into 3 of the 4 flows as a final block.

## Prerequisites

- SDG Hub installed: `pip install sdg-hub[examples]`
- vLLM serving gpt-oss-120b (teacher model) on an OpenAI-compatible endpoint
- Structured datasets from Step 1 in S3
- Environment variables configured (see [.env.example](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/.env.example))

## Procedure

### 2.1 Load Seed Data

```python
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()

seed_data_path = os.getenv("SEED_DATA_PATH", "seed_data.jsonl")
corpus = load_dataset("json", data_files=seed_data_path, split="train")

# Subsample for debugging (set SEED_DATA_SUBSAMPLE=0 to use full dataset)
subsample = int(os.getenv("SEED_DATA_SUBSAMPLE", "0"))
if subsample > 0:
    corpus = corpus.select(range(subsample))

corpus = corpus.to_pandas()
```

The seed data must have columns: `document`, `document_outline`, `domain` (produced by Step 1).

### 2.2 Discover Flows and Configure Model

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()
flows = FlowRegistry.list_flows()
print(f"Available flows: {flows}")
```

Configure the teacher model:

```python
def set_model_config(flow_object):
    flow_object.set_model_config(
        model=os.getenv("VLLM_MODEL", "hosted_vllm/gpt-oss-120b"),
        api_base=os.getenv("VLLM_API_BASE", "http://vllm-teacher:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    )
    return flow_object
```

### 2.3 Run All 4 Knowledge Tuning Flows

SDG Hub provides 4 knowledge tuning flows. Following the [worked example](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_generation.ipynb), all 4 are run and their outputs combined in the mixing step.

| Flow | ID | Output Type | Faithfulness Filter |
|------|-----|------------|-------------------|
| Extractive Summary Knowledge Tuning | `epic-jade-656` | Concise summaries + QA | Yes |
| Detailed Summary Knowledge Tuning | `mild-thunder-748` | Comprehensive summaries + QA | Yes |
| Key Facts Knowledge Tuning | `heavy-heart-77` | Atomic facts + 5 QA per fact | No |
| Document Based Knowledge Tuning | `stellar-peak-605` | Direct document + QA | Yes |

```python
number_of_summaries = int(os.getenv("NUMBER_OF_SUMMARIES", "50"))
max_concurrency = int(os.getenv("MAX_CONCURRENCY", "50"))
save_data_path = os.getenv("OUTPUT_DATA_FOLDER", "")

FLOW_NAMES = [
    "Extractive Summary Knowledge Tuning Dataset Generation Flow",
    "Detailed Summary Knowledge Tuning Dataset Generation Flow",
    "Key Facts Knowledge Tuning Dataset Generation Flow",
    "Document Based Knowledge Tuning Dataset Generation Flow",
]
```

**Flow 1: Extractive Summary**

```python
flow_path = FlowRegistry.get_flow_path(FLOW_NAMES[0])
flow = set_model_config(Flow.from_yaml(flow_path))

runtime_params = {"gen_extractive_summary": {"n": number_of_summaries}}
extractive_data = flow.generate(corpus, runtime_params=runtime_params, max_concurrency=max_concurrency)

extractive_data.to_json(
    os.path.join(save_data_path, "extractive_summary", "gen.jsonl"),
    orient="records", lines=True,
)
print(f"Extractive summary: {len(extractive_data)} records")
```

**Flow 2: Detailed Summary**

```python
flow_path = FlowRegistry.get_flow_path(FLOW_NAMES[1])
flow = set_model_config(Flow.from_yaml(flow_path))

runtime_params = {"gen_detailed_summary": {"n": number_of_summaries}}
detailed_data = flow.generate(corpus, runtime_params=runtime_params, max_concurrency=max_concurrency)

detailed_data.to_json(
    os.path.join(save_data_path, "detailed_summary", "gen.jsonl"),
    orient="records", lines=True,
)
print(f"Detailed summary: {len(detailed_data)} records")
```

**Flow 3: Key Facts**

```python
flow_path = FlowRegistry.get_flow_path(FLOW_NAMES[2])
flow = set_model_config(Flow.from_yaml(flow_path))

key_facts_data = flow.generate(corpus, max_concurrency=max_concurrency)

key_facts_data.to_json(
    os.path.join(save_data_path, "key_facts_to_qa", "gen.jsonl"),
    orient="records", lines=True,
)
print(f"Key facts: {len(key_facts_data)} records")
```

> **Note:** The Key Facts flow does not include faithfulness filtering and does not use `n` for summary count — it generates 5 QA pairs per atomic fact extracted from the document.

**Flow 4: Document Based**

```python
flow_path = FlowRegistry.get_flow_path(FLOW_NAMES[3])
flow = set_model_config(Flow.from_yaml(flow_path))

document_data = flow.generate(corpus, max_concurrency=max_concurrency)

document_data.to_json(
    os.path.join(save_data_path, "document_based_qa", "gen.jsonl"),
    orient="records", lines=True,
)
print(f"Document based: {len(document_data)} records")
```

### 2.4 Mix and Convert to Training Format

After generating all 4 outputs, combine them into a final training dataset using the [knowledge_mixing.ipynb](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/knowledge_mixing.ipynb) notebook. This step:

- Combines all 4 JSONL outputs
- Converts to OpenAI messages format (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`)
- Applies any additional curation

The final output is a single `sdg_output.jsonl` ready for Step 3.

## Output Format

Each line in the final JSONL file:

```json
{"messages": [{"role": "user", "content": "What should a NOC engineer do when..."}, {"role": "assistant", "content": "The engineer should first verify..."}]}
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| 4 intermediate JSONL files (one per flow) | S3 (`s3://noc-pipeline/sdg-output/`) | Mixing step |
| `sdg_output.jsonl` (combined, messages format) | S3 (`s3://noc-pipeline/sdg-output/`) | Step 3 (Model Training) |

## Next Step

[Step 3: Model Training →](step-3-model-training.md)
