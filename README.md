
# Advanced RAG Agent with Context Compression & Semantic Memory

An intelligent Retrieval-Augmented Generation (RAG) system combining hybrid search, parent-child document retrieval, Cohere reranking, and persistent semantic memory.

## Features

- Parent-child document retrieval
- Dense and sparse hybrid search
- Reciprocal Rank Fusion (RRF)
- Cohere-based document reranking
- Context compression
- Persistent semantic memory using Mem0
- Personalized responses based on user preferences
- FastAPI endpoints
- Retrieval, reranking, generation and total latency tracking

## Tech Stack

- Python
- FastAPI
- LangChain
- Qdrant
- OpenAI
- Cohere
- Mem0
- BM25
- Pytest
- Docker

## Architecture

Document Ingestion → Parent-Child Chunking → Qdrant Indexing

User Query → Dense + Sparse Retrieval → RRF Fusion
→ Parent Retrieval → Cohere Reranking
→ Context Compression → Semantic Memory
→ LLM Response

## Project Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd advanced-rag-memory-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file using `.env.example` and configure the required API keys.

### 6. Start Qdrant

```bash
docker compose up -d
```

### 7. Run the application

```bash
uvicorn app.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /api/chat | Chat with the RAG agent |
| GET | /api/users/{user_id}/memories | Retrieve user memories |

## Run Tests

```bash
pytest tests/ -v
```

## Performance Monitoring

The API tracks:

- retrieval_latency_ms
- rerank_latency_ms
- generation_latency_ms
- total_latency_ms

## Security

- API keys are stored in environment variables.
- `.env` is excluded from Git.
- `.env.example` contains placeholder values only.
