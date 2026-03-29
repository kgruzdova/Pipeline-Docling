"""Пайплайн индексации: DocLing → эмбеддинги → Pinecone."""

from __future__ import annotations

from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from loguru import logger

from hay_v2_bot.components.docling_ingest import attach_file_metadata, convert_path_to_documents
from hay_v2_bot.components.pinecone_store import PineconeDocumentStoreSafe


def run_ingestion_pipeline(
    *,
    local_path: str,
    user_id: int,
    filename: str,
    username: str | None,
    document_embedder: OpenAIDocumentEmbedder,
    document_store: PineconeDocumentStoreSafe,
) -> tuple[int, str]:
    """
    Анализирует файл через DocLing, обогащает meta, эмбеддит и пишет в Pinecone.
    Возвращает (число записанных чанков, объединённый текст для краткого резюме).
    """
    logger.info("ingestion: DocLing conversion path={}", local_path)
    raw_docs = convert_path_to_documents(local_path)
    if not raw_docs:
        logger.warning("ingestion: DocLing не вернул документов")
        return 0, ""

    preview_parts = [(d.content or "").strip() for d in raw_docs[:32]]
    combined_preview = "\n\n".join(p for p in preview_parts if p)

    docs = attach_file_metadata(
        raw_docs,
        user_id=user_id,
        filename=filename,
        username=username,
    )
    logger.info("ingestion: чанков после meta: {}", len(docs))

    embedded = document_embedder.run(documents=docs)["documents"]
    written = document_store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
    logger.info("ingestion: upsert в Pinecone: {}", written)
    return len(embedded), combined_preview
