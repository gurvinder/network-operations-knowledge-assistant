# Step 5: Model Registry and Serving

**Layer:** Fine-Tuning Layer
**Components:** RHOAI Model Registry, [KServe](https://github.com/kserve/kserve), [vLLM](https://github.com/vllm-project/vllm)

## Overview

After passing evaluation, the fine-tuned model is registered in the RHOAI Model Registry with full lineage metadata and deployed via KServe + vLLM as an OpenAI-compatible inference endpoint.

## Prerequisites

- RHOAI Model Registry configured on OpenShift
- KServe operator with vLLM runtime installed
- Checkpoint from Step 3 that passed evaluation in Step 4
- GPU resources: 1× A100 80GB (recommended for serving)

## Procedure

### 5.1 Register Model in RHOAI Model Registry

```python
registry_entry = {
    "name": "gpt-oss-20b-noc-tuned",
    "version": "1.0.0",
    "description": "GPT OSS 20B fine-tuned with OSFT for NOC knowledge assistance",
    "model_format": "safetensors",
    "framework": "vllm",
    "artifacts": {
        "model_uri": "s3://noc-pipeline/checkpoints/osft-noc-v1/",
    },
    "metadata": {
        "base_model": "openai/gpt-oss-20b",
        "training_algorithm": "osft",
        "training_data": "sdg-hub-knowledge-tuning",
        "training_samples": 2000,
        "evaluation_tool": "eval-hub",
        "evaluation_score": 0.91,
    }
}
```

### 5.2 Create KServe InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gpt-oss-20b-noc-tuned
  namespace: <your-namespace>
  annotations:
    serving.kserve.io/deploymentMode: Standard
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-runtime
      storageUri: s3://noc-pipeline/checkpoints/osft-noc-v1/
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "80Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "80Gi"
```

Apply and verify:

```bash
kubectl apply -f inferenceservice-noc.yaml
kubectl get inferenceservice gpt-oss-20b-noc-tuned -n <your-namespace>
```

### 5.3 Verify the Endpoint

```bash
curl -X POST http://gpt-oss-20b-noc-tuned-predictor.<your-namespace>.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-noc-tuned",
    "messages": [
      {"role": "user", "content": "What should I do when a BTS critical alarm is received?"}
    ]
  }'
```

### 5.4 Canary Deployment (Continuous Refresh)

When deploying a new model version, use canary traffic splitting:

```yaml
spec:
  predictor:
    canaryTrafficPercent: 10  # send 10% to new version
    model:
      storageUri: s3://noc-pipeline/checkpoints/osft-noc-v2/
```

Gradually increase traffic: 10% → 25% → 50% → 100%.

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| Registered model version | RHOAI Model Registry | Audit and lineage tracking |
| OpenAI-compatible endpoint | KServe InferenceService | Step 6 (RAG Pipeline), Demo |

## Next Step

[Step 6: RAG Pipeline →](step-6-rag-pipeline.md)
