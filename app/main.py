
from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.memories import router as memory_router

app = FastAPI(
    title="Advanced RAG Memory Agent",
    description=(
        "Hybrid RAG with Cohere Reranking "
        "and Mem0 Semantic Memory"
    ),
    version="1.0.0",
)


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": "Advanced RAG Memory Agent",
    }


app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"],
)

app.include_router(
    memory_router,
    prefix="/api",
    tags=["Memory"],
)
