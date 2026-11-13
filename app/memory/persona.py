
from app.memory.mem0_manager import Mem0MemoryManager


class PersonaManager:

    def __init__(self):

        self.memory = Mem0MemoryManager()

    def build_persona(
        self,
        user_id: str,
        query: str,
    ) -> str:
        """
        Build personalized system prompt context
        using persistent user memories.
        """

        memories = self.memory.search_memories(
            user_id=user_id,
            query=query,
            limit=10,
        )

        if not memories:
            return "No previous user preferences available."

        memory_facts = []

        for item in memories:

            fact = item.get("memory")

            if fact:
                memory_facts.append(f"- {fact}")

        if not memory_facts:
            return "No previous user preferences available."

        return (
            "USER PERSONALIZATION PROFILE:\n"
            "Use these preferences when relevant:\n"
            + "\n".join(memory_facts)
        )
