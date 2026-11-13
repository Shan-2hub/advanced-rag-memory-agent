
from fastapi import APIRouter, HTTPException

from app.memory.mem0_manager import Mem0MemoryManager

router = APIRouter()

memory = Mem0MemoryManager()


@router.get("/users/{user_id}/memories")
def get_user_memories(user_id: str):

    try:

        memories = memory.get_user_memories(user_id)

        return {
            "user_id": user_id,
            "count": len(memories),
            "memories": memories,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
