
from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.retrieval.hybrid_retriever import HybridRetriever
from app.memory.mem0_manager import Mem0MemoryManager
from app.memory.persona import PersonaManager
from app.agent.generator import ResponseGenerator
from app.utils.telemetry import Telemetry

router = APIRouter()

# Initialize reusable application components
retriever = HybridRetriever()
memory = Mem0MemoryManager()
persona_manager = PersonaManager()
generator = ResponseGenerator()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    telemetry = Telemetry()

    try:

        # 1. Retrieve relevant documents
        rag_results = retriever.retrieve(
            request.query
        )

        telemetry.retrieval_latency_ms = (
            rag_results["retrieval_latency_ms"]
        )

        telemetry.rerank_latency_ms = (
            rag_results["rerank_latency_ms"]
        )

        # 2. Retrieve user persona
        persona = persona_manager.build_persona(
            request.user_id,
            request.query,
        )

        # 3. Prepare context
        documents = rag_results["documents"]

        context = "\n\n".join(
            document.page_content
            for document in documents
        )


        if not context:
            context = "No relevant knowledge documents found."

        # 4. Generate personalized response
        import time

        generation_start = time.perf_counter()

        answer = generator.generate(
            query=request.query,
            context=context,
            persona=persona,
        )

        telemetry.generation_latency_ms = round(
            (time.perf_counter() - generation_start) * 1000,
            2,
        )

        # 5. Save conversation into Mem0
        memory.save_conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            user_message=request.query,
            assistant_message=answer,
        )

        # 6. Prepare sources
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "parent_id": doc.metadata.get("parent_id"),
                "relevance_score": doc.metadata.get(
                    "relevance_score"
                ),
            }
            for doc in documents
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata=telemetry.get_metadata(),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(error)}",
        )
