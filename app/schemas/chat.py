
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    query: str = Field(min_length=1)

    user_id: str = Field(min_length=1)

    session_id: str = Field(min_length=1)


class ChatResponse(BaseModel):

    answer: str

    sources: list[dict]

    metadata: dict
