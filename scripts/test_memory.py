
import time

from app.memory.mem0_manager import Mem0MemoryManager
from app.memory.persona import PersonaManager


def main():

    user_id = "usr_99"
    session_id = "session_001"

    memory = Mem0MemoryManager()

    # Session 1: Save user preferences
    print("\nSESSION 1: Saving user preferences\n")

    result = memory.save_conversation(
        user_id=user_id,
        session_id=session_id,
        user_message=(
            "Hello! I work in Python and prefer "
            "brief, code-only responses."
        ),
        assistant_message=(
            "Understood. I can tailor my coding "
            "responses to your preferences."
        ),
    )

    print("Mem0 submission:", result)

    # Mem0 Platform processes extracted memories asynchronously.
    print("\nWaiting for memory extraction...")

    saved_memories = []

    for attempt in range(15):

        time.sleep(2)

        saved_memories = memory.get_user_memories(
            user_id
        )

        if saved_memories:
            break

        print(f"Checking memory... {attempt + 1}/15")

    if not saved_memories:
        raise RuntimeError(
            "No memories found. Check Mem0 processing "
            "status and API usage."
        )

    print("\nStored memories:")

    for item in saved_memories:
        print("-", item.get("memory"))

    # Session 2: A completely new session
    print("\nSESSION 2: New conversation\n")

    persona = PersonaManager()

    profile = persona.build_persona(
        user_id=user_id,
        query="programming language and response style",
    )

    print("Retrieved persona:")
    print(profile)

    print("\nMemory validation completed.")


if __name__ == "__main__":
    main()
