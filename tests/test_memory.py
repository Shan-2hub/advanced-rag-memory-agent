
from unittest.mock import MagicMock

from app.memory.mem0_manager import Mem0MemoryManager
from app.memory.persona import PersonaManager


def test_save_conversation():

    manager = Mem0MemoryManager.__new__(
        Mem0MemoryManager
    )

    manager.client = MagicMock()

    manager.client.add.return_value = {
        "status": "PENDING",
        "event_id": "test-event",
    }

    result = manager.save_conversation(
        user_id="usr_99",
        session_id="session_001",
        user_message="I prefer Python.",
        assistant_message="Understood.",
    )

    assert result["status"] == "PENDING"

    manager.client.add.assert_called_once()


def test_get_user_memories():

    manager = Mem0MemoryManager.__new__(
        Mem0MemoryManager
    )

    manager.client = MagicMock()

    manager.client.get_all.return_value = {
        "results": [
            {"memory": "User prefers Python."}
        ]
    }

    results = manager.get_user_memories("usr_99")

    assert len(results) == 1

    assert results[0]["memory"] == (
        "User prefers Python."
    )


def test_persona_adaptation():

    persona = PersonaManager.__new__(
        PersonaManager
    )

    persona.memory = MagicMock()

    persona.memory.search_memories.return_value = [
        {"memory": "User prefers Python."},
        {"memory": "User prefers concise responses."},
    ]

    profile = persona.build_persona(
        user_id="usr_99",
        query="Write a merge sort routine",
    )

    assert "Python" in profile
    assert "concise responses" in profile


def test_empty_persona():

    persona = PersonaManager.__new__(
        PersonaManager
    )

    persona.memory = MagicMock()

    persona.memory.search_memories.return_value = []

    profile = persona.build_persona(
        user_id="new_user",
        query="Hello",
    )

    assert profile == (
        "No previous user preferences available."
    )
