"""Собранные зависимости рантайма бота."""

from __future__ import annotations

from dataclasses import dataclass

from haystack.components.agents import Agent
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from hay_v2_bot.components.pinecone_store import PineconeDocumentStoreSafe


@dataclass
class BotRuntime:
    document_store: PineconeDocumentStoreSafe
    text_embedder: OpenAITextEmbedder
    document_embedder: OpenAIDocumentEmbedder
    retriever: PineconeEmbeddingRetriever
    agent: Agent
