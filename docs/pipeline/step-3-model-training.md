# Step 3: Model Training

**Layer:** Fine-Tuning Layer
**Components:** [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) (OSFT), [Kubeflow Trainer v2](https://github.com/kubeflow/trainer)

## Overview

Training Hub fine-tunes GPT OSS 20B on the synthetic Q&A data using OSFT (Orthogonal Subspace Fine-Tuning). OSFT injects domain knowledge while preserving the base model's general capabilities. Training is orchestrated at scale via Kubeflow Trainer v2 TrainJob resources on OpenShift.

## Prerequisites

- Training Hub installed: `pip install training-hub`
- Kubeflow Trainer v2 operator deployed on OpenShift
- GPT OSS 20B base model accessible via HuggingFace token
- SDG output JSONL from Step 2 in S3
- GPU resources: 2× A100 80GB (recommended)

## Procedure

### 3.1 Configure OSFT Parameters

```python
osft_config = {
    "algorithm": "osft",
    "model_name_or_path": "openai/gpt-oss-20b",
    "unfreeze_rank_ratio": 0.25,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "bf16": True,
    "output_dir": "/output/osft-noc",
    "save_strategy": "epoch",
}
```

Key parameter: `unfreeze_rank_ratio` (0.2–0.3) controls what fraction of the model's subspace is updated. Higher values inject more knowledge but risk overwriting general capabilities.

### 3.2 Run Training Locally (Notebook)

```python
from training_hub import osft

osft(
    model_path="openai/gpt-oss-20b",
    data_path="s3://noc-pipeline/sdg-output/sdg_output.jsonl",
    unfreeze_rank_ratio=0.25,
    output_dir="/output/osft-noc",
)
```

### 3.3 Run Training at Scale (Kubeflow Trainer v2)

Create a TrainJob manifest for distributed training:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: noc-assistant-osft
  namespace: <your-namespace>
spec:
  runtimeRef:
    name: osft-clustertrainingruntime
  trainer:
    image: quay.io/rh-ai-quickstart/training-hub-osft:latest
    command: ["python", "-m", "training_hub.train_osft"]
    env:
      - name: MODEL_NAME
        value: "openai/gpt-oss-20b"
      - name: DATASET_PATH
        value: "/data/sdg_output.jsonl"
      - name: OUTPUT_DIR
        value: "/output/osft-noc"
      - name: UNFREEZE_RANK_RATIO
        value: "0.25"
    resourcesPerNode:
      requests:
        nvidia.com/gpu: "1"
        memory: "80Gi"
      limits:
        nvidia.com/gpu: "1"
        memory: "80Gi"
    numNodes: 2
  initializer:
    dataset:
      storageUri: s3://noc-pipeline/sdg-output/
```

Apply the manifest:

```bash
kubectl apply -f trainjob-noc-osft.yaml
kubectl get trainjob noc-assistant-osft -n <your-namespace> -w
```

### 3.4 Incremental Training (Continuous Refresh)

For subsequent training runs on new documents, resume from the latest checkpoint:

```python
osft(
    model_path="/output/osft-noc/hf_format/samples_2000.0",  # previous checkpoint
    data_path="s3://noc-pipeline/sdg-output/incremental_sdg.jsonl",
    unfreeze_rank_ratio=0.25,
    learning_rate=1e-5,       # lower LR for incremental
    num_train_epochs=1,        # fewer epochs
    output_dir="/output/osft-noc-v2",
)
```

## Output

The training produces a HuggingFace-format checkpoint:

```
/output/osft-noc/hf_format/samples_2000.0/
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── model.safetensors.index.json
└── model-00001-of-00008.safetensors
    ...
```

| Output | Destination | Consumed By |
|--------|------------|-------------|
| safetensors checkpoint | S3 (`s3://noc-pipeline/checkpoints/`) | Step 4 (Evaluation) |

## Next Step

[Step 4: Evaluation →](step-4-evaluation.md)
