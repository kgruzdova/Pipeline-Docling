"""Pinecone document store с безопасной подготовкой metadata (без мутации Document)."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any

from haystack import Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone.document_store import (
    METADATA_SUPPORTED_TYPES,
    PineconeDocumentStore,
)
from loguru import logger

from hay_v2_bot.config import PINECONE_DIMENSION, PINECONE_INDEX, PINECONE_NAMESPACE


def _metadata_value_ok(value: Any) -> bool:
    return isinstance(value, METADATA_SUPPORTED_TYPES) or (
        isinstance(value, list) and all(isinstance(i, str) for i in value)
    )


def document_with_sanitized_meta(document: Document) -> Document:
    if not document.meta:
        return document
    discarded_keys: list[str] = []
    new_meta: dict[str, Any] = {}
    for key, value in document.meta.items():
        if not _metadata_value_ok(value):
            discarded_keys.append(key)
        else:
            new_meta[key] = value
    if discarded_keys:
        logger.warning(
            "Document {}: отброшены поля meta неподдерживаемых типов {}.",
            document.id,
            discarded_keys,
        )
    return replace(document, meta=new_meta)


class PineconeDocumentStoreSafe(PineconeDocumentStore):
    def _convert_documents_to_pinecone_format(
        self, documents: list[Document]
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        logger.info(
            "run _convert_documents_to_pinecone_format: подготовка {} документов к записи в Pinecone",
            len(documents),
        )
        documents_for_pinecone: list[tuple[str, list[float], dict[str, Any]]] = []
        for document in documents:
            doc = document_with_sanitized_meta(document) if document.meta else document
            embedding = copy(doc.embedding)
            if embedding is None:
                logger.warning(
                    "Document {}: нет embedding; подставляется dummy-вектор.",
                    doc.id,
                )
                embedding = self._dummy_vector

            metadata = dict(doc.meta) if doc.meta else {}
            if doc.content is not None:
                metadata["content"] = doc.content
            if doc.blob is not None:
                logger.warning("Document {}: поле blob не сохраняется в Pinecone.", doc.id)

            documents_for_pinecone.append((doc.id, embedding, metadata))
        return documents_for_pinecone


def build_document_store() -> PineconeDocumentStoreSafe:
    return PineconeDocumentStoreSafe(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=PINECONE_INDEX,
        namespace=PINECONE_NAMESPACE,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
    )
