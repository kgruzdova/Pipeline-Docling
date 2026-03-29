"""Долговременная память: извлечение контекста и сохранение реплик пользователя."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone.document_store import PineconeDocumentStore
from loguru import logger

from hay_v2_bot.config import MEMORY_TOP_K, META_TYPE_FILE_CHUNK, META_TYPE_USER_MESSAGE


def retrieve_memory_context(
    user_id: int,
    query: str,
    text_embedder: OpenAITextEmbedder,
    retriever: PineconeEmbeddingRetriever,
) -> str:
    q_preview = query if len(query) <= 400 else query[:400] + "..."
    logger.info("ContextRetriever.run: поиск. query: {}", q_preview)
    emb = text_embedder.run(text=query)["embedding"]
    logger.info("ContextRetriever.run: эмбеддинг создан, размер: {}", len(emb))
    flt = {"field": "user_id", "operator": "==", "value": str(user_id)}
    docs = retriever.run(query_embedding=emb, filters=flt, top_k=MEMORY_TOP_K)["documents"]
    logger.info("ContextRetriever.run: найдено документов: {}", len(docs))
    if not docs:
        return ""
    lines = []
    for d in docs:
        content = (d.content or "").strip()
        if content:
            meta = d.meta or {}
            prefix = ""
            mtype = meta.get("type")
            if mtype == META_TYPE_USER_MESSAGE:
                prefix = "[сообщение] "
            elif mtype == META_TYPE_FILE_CHUNK:
                fname = meta.get("filename") or "файл"
                prefix = f"[фрагмент: {fname}] "
            else:
                prefix = ""
            lines.append(f"- {prefix}{content}")
    return "\n".join(lines)


def persist_turn(
    user_id: int,
    user_text: str,
    doc_embedder: OpenAIDocumentEmbedder,
    store: PineconeDocumentStore,
    username: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    uid = str(user_id)
    meta_user = {
        "user_id": uid,
        "type": META_TYPE_USER_MESSAGE,
        "timestamp": ts,
        "username": username or "",
    }
    docs = [
        Document(
            id=str(uuid.uuid4()),
            content=user_text,
            meta={**meta_user, "role": "user"},
        ),
    ]
    embed_out = doc_embedder.run(documents=docs)
    with_embeddings = embed_out["documents"]
    written = store.write_documents(with_embeddings, policy=DuplicatePolicy.OVERWRITE)
    logger.info("ContextSaver.run: записано в Pinecone: {}", written)


def forget_user_memory(store: PineconeDocumentStore, user_id: int) -> int:
    flt = {"field": "user_id", "operator": "==", "value": str(user_id)}
    return store.delete_by_filter(flt)
