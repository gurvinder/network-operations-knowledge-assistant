# Step 6: RAG Pipeline and Demo

**Layer:** Inference Layer
**Components:** [PGVector](https://github.com/pgvector/pgvector), OGX/Llama Stack, Ingestion Pipeline, [Streamlit](https://github.com/streamlit/streamlit)
**Source:** [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts)

## Overview

The Inference Layer combines the fine-tuned model (from Step 5) with the embeddings (from Step 1) to deliver RAG-augmented responses. OGX/Llama Stack orchestrates the query pipeline: retrieving relevant document chunks from PGVector and augmenting prompts to the fine-tuned model. A Streamlit chat application provides the demo interface.

## Prerequisites

- PGVector populated with embeddings from Step 1
- Fine-tuned model deployed via KServe from Step 5
- Helm charts from `rh-ai-quickstart/ai-architecture-charts`

## Procedure

### 6.1 Deploy RAG Infrastructure via Helm

```bash
# Clone the charts
git clone https://github.com/rh-ai-quickstart/ai-architecture-charts.git
cd ai-architecture-charts

# Deploy PGVector (if not already deployed in Step 1)
helm install pgvector ./pgvector/helm

# Deploy OGX / Llama Stack for RAG orchestration
helm install ogx-ai ./ogx-ai/helm \
  --set llm.endpoint=http://gpt-oss-20b-noc-tuned.<your-namespace>.svc/v1 \
  --set vectorDb.host=pgvector.<your-namespace>.svc \
  --set vectorDb.port=5432

# Deploy the ingestion pipeline
helm install ingestion-pipeline ./ingestion-pipeline/helm \
  --set defaultPipeline.enabled=true \
  --set defaultPipeline.source=S3
```

### 6.2 Verify RAG Query Pipeline

The RAG flow: **Query → Retrieve from PGVector → Augment prompt → Generate answer**

```bash
curl -X POST http://ogx-ai.<your-namespace>.svc/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-noc-tuned",
    "messages": [
      {"role": "user", "content": "How do I troubleshoot a BTS critical failure?"}
    ]
  }'
```

The response includes both the answer and source citations:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Based on the NOC runbook for BTS failures, you should first..."
    }
  }],
  "sources": [
    {
      "document": "noc-runbook-example.md",
      "section": "Immediate Actions > Step 1",
      "score": 0.92,
      "text": "Check NMS for correlated alarms..."
    }
  ]
}
```

### 6.3 Deploy Streamlit Demo

```bash
helm install streamlit-demo ./streamlit/helm \
  --set ragEndpoint=http://ogx-ai.<your-namespace>.svc/v1
```

The demo provides:
- Chat interface for NOC engineers
- Answers with inline source citations
- Retrieved passage highlights with relevance scores
- Document and section references for audit trail

### 6.4 Verify End-to-End

1. Open the Streamlit UI at the deployed route
2. Ask: "What are the escalation criteria for a BTS failure?"
3. Verify the response includes:
   - Accurate answer grounded in the NOC runbook
   - Source citations with document name and section
   - Relevance scores for retrieved passages

## Architecture Flow

```
User Query
    ↓
OGX / Llama Stack
    ↓
PGVector (retrieve relevant chunks from Data Layer embeddings)
    ↓
Augment prompt with retrieved context
    ↓
KServe + vLLM (fine-tuned GPT OSS 20B)
    ↓
Answer + Source Citations
    ↓
Streamlit Chat UI
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| RAG-augmented responses | OGX API endpoint | Streamlit Demo, API consumers |
| Chat UI | Streamlit route | NOC engineers |

## Previous Steps

- [← Step 5: Model Registry and Serving](step-5-model-serving.md)
- [← Step 1: Data Preparation](step-1-data-preparation.md) (embeddings source)
- [Overview](00-overview.md)
