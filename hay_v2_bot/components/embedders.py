"""OpenAI text/document embedders: только HTTP API (в т.ч. прокси), без локальных эмбеддингов."""

from __future__ import annotations

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret

from hay_v2_bot.config import EMBEDDING_MODEL, OPENAI_API_BASE_URL, PINECONE_DIMENSION


def build_embedders() -> tuple[OpenAITextEmbedder, OpenAIDocumentEmbedder]:
    common = dict(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL,
        dimensions=PINECONE_DIMENSION,
        api_base_url=OPENAI_API_BASE_URL,
    )
    text_e = OpenAITextEmbedder(**common)
    doc_e = OpenAIDocumentEmbedder(**common)
    return text_e, doc_e
