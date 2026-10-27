# Intelligent Knowledge Base Retrieval System

A domain-specific retrieval and QA system built on **LangGraph + Milvus + BGE-M3**, covering the full pipeline from PDF ingestion to citation-grounded answers.

> 中文文档见 [README.md](README.md)

---

## Features

- **Ingestion pipeline** — PDF → layout parsing → image extraction → semantic chunking → embedding → Milvus
- **Hybrid recall** — dense semantic recall (BGE-M3) fused with sparse keyword recall via Reciprocal Rank Fusion
- **Re-ranking** — cross-encoder re-ranker refines the Top-K candidates
- **Grounded answers** — every answer cites its source passages for traceability
- **Session memory** — conversation history persisted in MongoDB
- **Streaming** — SSE-based token streaming

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.2 / LangChain |
| Embedding | BGE-M3 (`flagembedding`) |
| Vector DB | Milvus 2.4 standalone |
| Object storage | MinIO |
| Document store | MongoDB |
| PDF parsing | MinerU |
| Re-ranking | Qwen Rerank via HTTP |
| Web | FastAPI + Uvicorn |

---

## Quick start

```bash
pip install -r requirements.txt

cp .env.example .env      # fill in LLM / Milvus / MinIO / MongoDB / Rerank config

python -m web.api.query_service      # query service, port 8001
python -m web.api.import_service     # ingestion service, port 8002
```

See [`docs/ENV_AND_DEPS.md`](docs/ENV_AND_DEPS.md) for the full environment reference.

---

## License

[MIT](LICENSE)
