# Step 6: RAG Pipeline and Demo

**Layer:** Inference Layer
**Components:** [PGVector](https://github.com/pgvector/pgvector), [OGX](https://github.com/rh-ai-quickstart/ai-architecture-charts/tree/main/ogx-ai), Ingestion Pipeline, RHOAI Gen AI Studio
**Source:** [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts)

## Overview

The Inference Layer combines the fine-tuned model (from Step 5) with the embeddings (from Step 1) to deliver RAG-augmented responses. OGX orchestrates the query pipeline using the `/v1/responses` API with `file_search`: retrieving relevant document chunks from PGVector and augmenting prompts to the fine-tuned model. The RHOAI Gen AI Studio (Playground) provides the demo interface.

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

# Deploy MinIO for document storage
helm install minio ./minio/helm --set sampleFileUpload.enabled=true

# Deploy OGX with the fine-tuned model enabled
helm install ogx-ai ./ogx-ai/helm \
  --set models.gpt-oss-20b.enabled=true

# Deploy the ingestion pipeline (uses OGX as gateway)
helm install ingestion-pipeline ./ingestion-pipeline/helm \
  --set defaultPipeline.enabled=true \
  --set defaultPipeline.source=S3 \
  --set defaultPipeline.S3.bucket_name=documents \
  --set defaultPipeline.S3.endpoint_url=http://minio:9000 \
  --set clientDependency=ogx-ai

# Deploy the demo UI (RHOAI Gen AI Studio/Playground)
helm install playground ./playground/helm
```

### 6.2 Set Up Vector Store and Upload Documents

OGX manages RAG via the OpenAI-compatible Files and Vector Stores APIs:

```python
from openai import OpenAI

client = OpenAI(base_url="http://ogx-ai.<your-namespace>.svc:8321/v1/", api_key="none")

# Create a vector store for NOC knowledge
vs = client.vector_stores.create(name="noc-knowledge-base")

# Upload and attach a document
file = client.files.create(
    file=open("noc-runbook.pdf", "rb"),
    purpose="assistants",
)
client.vector_stores.files.create(
    vector_store_id=vs.id,
    file_id=file.id,
)
```

OGX automatically chunks, embeds, and indexes the uploaded documents into PGVector.

### 6.3 Query via the Responses API

The RAG flow uses `/v1/responses` with `file_search` (not `/v1/chat/completions`):

```python
response = client.responses.create(
    model="gpt-oss-20b",
    input="What is the procedure for BGP flap recovery?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vs.id],
    }],
    include=["file_search_call.results"],
)

# Get the answer
print(response.output_text)

# Get retrieved source chunks with scores
for item in response.output:
    if item.type == "file_search_call" and item.results:
        for result in item.results:
            print(f"  Source: {result.filename} (score: {result.score})")
            print(f"  Text: {result.text[:200]}...")
```

**cURL equivalent:**

```bash
curl -X POST http://ogx-ai.<your-namespace>.svc:8321/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "input": "How do I troubleshoot a BTS critical failure?",
    "tools": [{
      "type": "file_search",
      "vector_store_ids": ["vs_abc123"]
    }],
    "include": ["file_search_call.results"]
  }'
```

### 6.4 Response Format

The response includes both the answer and retrieved source chunks:

```json
{
  "id": "resp_abc123",
  "output": [
    {
      "type": "file_search_call",
      "id": "fs_abc123",
      "status": "completed",
      "queries": ["BTS critical failure troubleshooting"],
      "results": [
        {
          "file_id": "file-xyz789",
          "filename": "noc-runbook-example.md",
          "text": "When a BTS reports a critical alarm, first verify alarm authenticity...",
          "score": 0.92
        }
      ]
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The procedure for BTS critical failure troubleshooting involves...",
          "annotations": [
            {
              "type": "file_citation",
              "file_id": "file-xyz789",
              "filename": "noc-runbook-example.md",
              "index": 47
            }
          ]
        }
      ]
    }
  ]
}
```

### 6.5 Deploy Demo UI

The RHOAI Gen AI Studio (Playground) is deployed via the `playground` chart, which registers your OGX endpoint, vector stores, and MCP servers with the RHOAI dashboard:

```bash
helm install playground ./playground/helm
```

The playground provides:
- Chat interface for NOC engineers
- Answers with inline source citations (file_citation annotations)
- Integration with registered vector stores and MCP tool servers
- Model selection across deployed endpoints

### 6.6 Verify End-to-End

1. Open the RHOAI Gen AI Studio at the deployed route
2. Select the `gpt-oss-20b` model endpoint
3. Ask: "What are the escalation criteria for a BTS failure?"
4. Verify the response includes:
   - Accurate answer grounded in the NOC knowledge base
   - Source citations with document name
   - Retrieved passage scores

## Architecture Flow

```
User Query
    |
OGX (/v1/responses with file_search)
    |
PGVector (retrieve relevant chunks from Data Layer embeddings)
    |
Augment prompt with retrieved context
    |
KServe + vLLM (fine-tuned GPT OSS 20B)
    |
Answer + file_citation annotations
    |
RHOAI Gen AI Studio (Playground)
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| RAG-augmented responses | OGX `/v1/responses` endpoint | Playground, API consumers |
| Demo UI | RHOAI Gen AI Studio route | NOC engineers |

## Previous Steps

- [Step 5: Model Registry and Serving](step-5-model-serving.md)
- [Step 1: Data Preparation](step-1-data-preparation.md) (embeddings source)
- [Overview](00-overview.md)
