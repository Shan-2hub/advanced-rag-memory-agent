
import os

from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv()


class Mem0MemoryManager:

    def __init__(self):

        api_key = os.getenv("MEM0_API_KEY")

        if not api_key:
            raise ValueError("MEM0_API_KEY is missing.")

        self.client = MemoryClient(api_key=api_key)

    def save_conversation(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ):
        """
        Extract and save important user facts
        from a conversation.
        """

        messages = [
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": assistant_message,
            },
        ]

        result = self.client.add(
            messages,
            user_id=user_id,
            run_id=session_id,
            metadata={
                "source": "advanced_rag_agent",
            },
        )

        return result

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ):
        """
        Retrieve relevant memories for a user.
        """

        result = self.client.search(
            query,
            filters={"user_id": user_id},
            top_k=limit,
        )

        return result.get("results", [])

    def get_user_memories(self, user_id: str):
        """
        Retrieve all stored memories for a user.
        """

        result = self.client.get_all(
            filters={"user_id": user_id},
            page=1,
            page_size=100,
        )

        return result.get("results", [])
