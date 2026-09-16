# Step 1: Data Preparation

**Layer:** Data Layer
**Components:** [Docling](https://github.com/docling-project/docling), [GSMA Datasets](https://huggingface.co/GSMA/datasets)

## Overview

This step parses raw documents into structured representations that feed both the Fine-Tuning and Inference layers. Documents are processed once through Docling and produce two outputs: structured datasets (for SDG) and token-aware chunks (for embedding).

## Prerequisites

- Python 3.11+
- Docling installed: `pip install docling docling-core`
- Access to GSMA Datasets on HuggingFace (or local document corpus)
- S3-compatible storage (MinIO) for output datasets

## Input Sources

| Source | Format | Access |
|--------|--------|--------|
| [GSMA/3GPP](https://huggingface.co/datasets/GSMA/3GPP) | HuggingFace Dataset | `datasets.load_dataset("GSMA/3GPP")` |
| [GSMA/etsi](https://huggingface.co/datasets/GSMA/etsi) | HuggingFace Dataset | `datasets.load_dataset("GSMA/etsi")` |
| [GSMA/oran](https://huggingface.co/datasets/GSMA/oran) | HuggingFace Dataset | `datasets.load_dataset("GSMA/oran")` |
| [GSMA/itu](https://huggingface.co/datasets/GSMA/itu) | HuggingFace Dataset | `datasets.load_dataset("GSMA/itu")` |
| [GSMA/Telco-Common-Corpus](https://huggingface.co/datasets/GSMA/Telco-Common-Corpus) | HuggingFace Dataset | 1.78M documents across all telecom standards |
| Products and Releases Documentation | PDF, DOCX, HTML | Proprietary operator-specific runbooks and vendor manuals |

## Procedure

### 1.1 Parse Documents with Docling

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("path/to/document.pdf")
doc = result.document
```

Docling supports 15+ input formats (PDF, DOCX, HTML, Markdown, PPTX, etc.) and produces a structured `DoclingDocument` with section headers, tables, and metadata preserved.

### 1.2 Generate Structured Datasets (for FT Layer)

Use `HierarchicalChunker` to preserve full document sections with structural outlines:

```python
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import DocItemLabel

chunker = HierarchicalChunker()
chunks = list(chunker.chunk(doc))

# Extract structured markdown
document_text = doc.export_to_markdown()

# Extract document outline from headings
outline_parts = []
for item, level in doc.iterate_items():
    if hasattr(item, 'label') and item.label in (
        DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE
    ):
        text = item.text if hasattr(item, 'text') else str(item)
        outline_parts.append(f"{'#' * (level + 1)} {text}")
document_outline = "\n".join(outline_parts)
```

### 1.3 Build SDG Hub Input Dataset

Map Docling output directly to the SDG Hub input schema:

```python
from datasets import Dataset

rows = [{
    "document": document_text,
    "document_outline": document_outline,
    "domain": "telecom_noc_alarm_triage",  # derived from source category
}]

ds = Dataset.from_list(rows)
ds.save_to_disk("s3://noc-pipeline/datasets/structured/")
```

### 1.4 Generate Token-Aware Chunks (for Inference Layer)

Use `HybridChunker` for embedding-ready chunks:

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

hf_tokenizer = HuggingFaceTokenizer(
    tokenizer=my_tokenizer,  # GSMA OTel embedding tokenizer
    max_tokens=512,
)
chunker = HybridChunker(
    tokenizer=hf_tokenizer,
    merge_peers=True,
)
chunks = list(chunker.chunk(doc))
```

### 1.5 Embed and Store in PGVector

Embed chunks using GSMA OTel embedding models and store in PGVector:

```python
import hashlib

for i, chunk in enumerate(chunks):
    text = chunk.text if hasattr(chunk, 'text') else str(chunk)
    embedding = embed_model.encode(text)  # GSMA OTel model

    insert_row = {
        "content": text,
        "embedding": embedding,
        "document_name": doc.name,
        "chunk_index": i,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
    }
    # Insert into PGVector
```

## Output

| Output | Destination | Consumed By |
|--------|------------|-------------|
| Structured datasets (MD + outlines + domain) | S3 (`s3://noc-pipeline/datasets/structured/`) | Step 2 (SDG Hub) |
| Embedded chunks | PGVector | Step 6 (RAG Pipeline) |

## Next Step

[Step 2: Synthetic Data Generation →](step-2-synthetic-data-generation.md)
