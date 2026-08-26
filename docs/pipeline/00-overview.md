# Network Operations Knowledge Assistant Pipeline

## Purpose

An end-to-end pipeline that builds a **Network Operations Knowledge Assistant** — an AI assistant that helps telecom engineers troubleshoot network issues, reference NOC runbooks, and follow incident response procedures using accurate, domain-specific answers.

The assistant is built by extracting knowledge from internal operations documents (runbooks, equipment manuals, incident playbooks), generating synthetic training data with **SDG Hub**, fine-tuning an open-source model with **OSFT** via **Training Hub** and **Kubeflow Trainer**, and serving the tuned model with **RAG** for grounded, source-backed responses. All components run on Red Hat OpenShift AI.

**Industry:** Telecommunications — Mobile Operators (GSMA audience)

**Intended Audience:**
- Mobile operators and telecom companies (GSMA members)
- Enterprise teams looking to build domain-specific AI assistants from internal knowledge
- ML platform teams evaluating alternatives to proprietary LLM APIs

**Reference Implementations:**
- [Knowledge tuning end-to-end example (InstructLab methodology)](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/knowledge-tuning)
- [SDG Hub knowledge tuning flow and examples](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/tree/main/examples/knowledge_tuning)
- [SDG Hub documentation](https://ai-innovation.team/sdg_hub/)

## Architecture Diagram

![Pipeline Architecture](images/pipeline-architecture.svg)

## Component Selection

| Step | Component | Source | Rationale |
|------|-----------|--------|-----------|
| 1. Data Preparation | [Docling](https://github.com/docling-project/docling) | IBM / LF AI & Data | Structure-aware parsing of PDF, DOCX, HTML; dual chunking for SDG and RAG paths |
| 2. Taxonomy Curation | [InstructLab Taxonomy](https://github.com/instructlab/taxonomy) | RHOAI | Natural input format for SDG Hub; organizes domain knowledge into seed datasets |
| 3. Synthetic Data Generation | [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) + [vLLM](https://github.com/vllm-project/vllm) Teacher | RHOAI | Composable YAML flows, knowledge tuning with faithfulness filtering |
| 4. Model Training | [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) (OSFT) + [Kubeflow Trainer v2](https://github.com/kubeflow/trainer) | RHOAI | Knowledge injection without catastrophic forgetting; distributed training orchestration |
| 5. Evaluation | Comparative Eval (Human Rubric) | Custom | Side-by-side scoring: base model vs tuned vs tuned+RAG |
| 6. Model Registry & Serving | RHOAI Model Registry + [KServe](https://github.com/kserve/kserve) + [vLLM](https://github.com/vllm-project/vllm) | RHOAI | Native registry workflow, serverless autoscaling, OpenAI-compatible API |
| 7. RAG + Serving | [PGVector](https://github.com/pgvector/pgvector) + OGX/Llama Stack + Ingestion Pipeline | [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts) | Helm-deployed RAG stack; augments fine-tuned model with new docs post-training |
| Demo | [Streamlit](https://github.com/streamlit/streamlit) (via ai-architecture-charts) | [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts) | Deployed alongside the RAG stack; chat UI with source citations |

## Architecture Summary

The solution follows the 4-stage architecture described in the [design issue](https://github.com/rh-ai-quickstart/ai-quickstart-contrib/issues/73), expanded into actionable pipeline steps:

### Stage 1: Data Processing (Steps 1–2)

Documents enter the pipeline from three source categories — synthetic NOC runbook templates (covering 5 operational domains), 3GPP/ETSI public specifications, and O-RAN Alliance documentation. **Docling** parses these into structured `DoclingDocument` representations with section headers, tables, and metadata preserved.

The parsed documents follow two parallel paths:

- **SDG Path:** The `HierarchicalChunker` preserves full document sections. Domain experts organize these into **InstructLab Taxonomy** seeds — structured `qna.yaml` files under 5 knowledge domains (Alarm Triage, IP/Transport, RAN Operations, Core Network, Escalation Procedures). Each seed provides document text, structural outlines, and domain labels.
- **RAG Path:** The `HybridChunker` produces token-aware chunks for downstream embedding and retrieval.

### Stage 2: Knowledge Generation (Step 3)

**SDG Hub** consumes the taxonomy seeds using the Key Facts flow for initial bootstrap and the Extractive Summary Knowledge Tuning flow for full-scale generation. A **vLLM Teacher Model** (gpt-oss-120b) generates ~2,000 synthetic Q&A pairs with faithfulness and relevancy scoring. Quality-filtered outputs are formatted as OpenAI-format messages JSONL.

### Stage 3: Model Training (Step 4)

**Training Hub** fine-tunes **GPT OSS 20B** on the synthetic data using **OSFT** (Orthogonal Subspace Fine-Tuning), which injects domain knowledge while preserving the base model's general capabilities. Training is orchestrated at scale via **Kubeflow Trainer v2** TrainJob resources on OpenShift.

### Stage 4: Serving & RAG (Steps 5–8)

**Comparative Evaluation** scores model outputs across three configurations (base model, OSFT-tuned, OSFT-tuned + RAG) using a structured rubric assessed by an evaluator persona. On passing evaluation, the model is registered in the **RHOAI Model Registry** and deployed via **KServe + vLLM** as an OpenAI-compatible endpoint.

The RAG pipeline is deployed via the **Ingestion Pipeline** chart (from [`rh-ai-quickstart/ai-architecture-charts`](https://github.com/rh-ai-quickstart/ai-architecture-charts)), which handles chunking, embedding generation, and storage into **PGVector**. **OGX** (or Llama Stack) orchestrates the RAG query pipeline — retrieving relevant document chunks and augmenting prompts to the fine-tuned model for grounded, source-cited responses.

A **Streamlit** chat application (deployed via the ai-architecture-charts pattern) provides the demo interface, displaying the assistant's answers alongside retrieved source passages and document citations.

## Knowledge Domains

| Domain | Sample Sources | Coverage |
|--------|---------------|----------|
| Alarm Triage & Incident Response | NOC playbook templates, [3GPP TS 32.111-2 Fault Management](https://www.3gpp.org/DynaReport/32111-2.htm) ([PDF](https://www.etsi.org/deliver/etsi_TS/132100_132199/13211102/19.00.00_60/ts_13211102v190000p.pdf)) | Alarm correlation, severity classification, initial diagnostics |
| IP/Transport Troubleshooting | NOC runbooks, [ETSI GS NFV 006 — NFV MANO Architecture](https://www.etsi.org/deliver/etsi_gs/NFV/001_099/006/05.02.01_60/gs_NFV006v050201p.pdf), [ETSI GR NFV-MAN 001](https://docbox.etsi.org/isg/nfv/open/Publications_pdf/Specs-Reports/NFV-MAN%20001v1.2.1%20-%20GR%20-%20Management%20and%20Orchestration.pdf) | BGP flap recovery, MPLS LSP repair, link failure procedures |
| RAN Operations (O-RAN) | [O-RAN WG10 OAM Architecture](https://specifications.o-ran.org/download?id=690), [O-RAN Specifications Portal](https://www.o-ran.org/specifications), [O-RAN SC O1 Data Models](https://lf-o-ran-sc.atlassian.net/wiki/display/ORAN/O1-Interface+Data+Models) | Cell outage recovery, interference management, RIC troubleshooting |
| Core Network (5GC/EPC) | [3GPP TS 23.501 5G System Architecture](https://www.3gpp.org/DynaReport/23501.htm), [3GPP TS 32.111-6 Alarm IRP Solution Sets](https://www.etsi.org/deliver/etsi_TS/132100_132199/13211106/19.00.00_60/ts_13211106v190000p.pdf) | Signaling failures, subscriber service recovery, element failover |
| Escalation & Vendor Coordination | NOC procedure templates (synthetic) | Tiered escalation, vendor RMA, change management approval |

## Shared Infrastructure

Deployment of the RAG and demo layers uses Helm charts from [`rh-ai-quickstart/ai-architecture-charts`](https://github.com/rh-ai-quickstart/ai-architecture-charts). This provides production-ready charts for:

| Component | Chart | Purpose |
|-----------|-------|---------|
| OpenShift + RHOAI | Platform | Platform layer |
| NVIDIA GPU Operator | Platform | GPU access for Steps 3, 4, 6, 7 |
| MinIO | `minio` | S3-compatible object storage for datasets, checkpoints, model artifacts |
| PGVector | `pgvector` | Vector database for RAG retrieval (pgvector extension) |
| LLM Service | `llm-service` | vLLM-based model serving with OpenAI-compatible API |
| OGX / Llama Stack | `ogx-ai` or `llama-stack` | RAG orchestration, agent capabilities, safety shields |
| Ingestion Pipeline | `ingestion-pipeline` | Document chunking, embedding, and vector store ingestion |
| Kubeflow Pipelines | Platform | Pipeline orchestration (production mode) |
| HuggingFace Token | Secret | Access to base model (GPT OSS 20B) and teacher model |

## Model Configuration

| Role | Model | Parameters | Notes |
|------|-------|-----------|-------|
| Teacher (SDG) | gpt-oss-120b | 120B | Default for SDG Hub knowledge tuning flows |
| Student (Fine-tuned) | GPT OSS 20B | 20B | OpenAI open-source model; fine-tuned with OSFT (`unfreeze_rank_ratio=0.2–0.3`) |
| Embedding | Granite Embedding / BGE-M3 | ~110M–560M | Served via vLLM `--task embed` |

## GPU Hardware Configurations

Configurations below are sized for the models specified in Model Configuration above.

| Profile | Training (OSFT) | Serving (vLLM) | Embedding | Notes |
|---------|----------------|----------------|-----------|-------|
| A100 80GB | 2× A100 80GB | 1× A100 80GB | 1× T4/L4 | Recommended for production |
| A100 40GB | 4× A100 40GB | 2× A100 40GB | 1× T4/L4 | Alternative with more GPUs |
| L40S 48GB | 2× L40S | 2× L40S | 1× L40S | Cost-effective alternative |

## Licensing

| Component | License |
|-----------|---------|
| Docling | MIT |
| InstructLab Taxonomy | Apache 2.0 |
| SDG Hub | Apache 2.0 |
| vLLM | Apache 2.0 |
| Training Hub | Apache 2.0 |
| Kubeflow Trainer | Apache 2.0 |
| PGVector | PostgreSQL License |
| OGX / Llama Stack | MIT |
| KServe | Apache 2.0 |
| Ingestion Pipeline | Apache 2.0 |
| Streamlit | Apache 2.0 |
| ai-architecture-charts | Apache 2.0 |
| RHOAI | Red Hat Subscription |

