
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class ResponseGenerator:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4.1-mini"
            ),
            temperature=0.2,
        )

    def generate(
        self,
        query: str,
        context: str,
        persona: str,
    ) -> str:

        system_prompt = f"""
You are an accurate Retrieval Augmented Generation assistant.

PERSONALIZATION:
{persona}

KNOWLEDGE CONTEXT:
{context}

INSTRUCTIONS:
1. Answer using the supplied knowledge context.
2. Do not invent facts that are not supported by context.
3. If the answer is unavailable, clearly say so.
4. Follow relevant user preferences.
5. Be concise and accurate.
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ])

        return response.content
