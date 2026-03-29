"""DocLing: конвертация файла → Haystack Document (чанки) без docling-haystack."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from haystack import Document
from loguru import logger

from hay_v2_bot.config import DOCLING_CHUNK_MAX_TOKENS, EMBEDDING_MODEL, META_TYPE_FILE_CHUNK


def _build_hybrid_chunker():
    """
    Разбиение на чанки по tiktoken (как у OpenAI text-embedding-3-small), без HF/sentence-transformers.
    Это не эмбеддинги — только подсчёт токенов для границ чанков; векторы считает OpenAIDocumentEmbedder.
    """
    import tiktoken
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    try:
        enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
        logger.warning(
            "tiktoken: для модели {} нет mapping, используется cl100k_base",
            EMBEDDING_MODEL,
        )

    max_tok = max(256, min(DOCLING_CHUNK_MAX_TOKENS, 8191))
    tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=max_tok)
    return HybridChunker(tokenizer=tokenizer)


def _page_from_dl_meta(dl_meta: Any) -> int | None:
    if dl_meta is None:
        return None
    try:
        from docling.chunking import DocChunk

        chunk = DocChunk.model_validate(dl_meta)
        items = chunk.meta.doc_items
        if not items:
            return None
        prov = items[0].prov
        if not prov:
            return None
        return int(prov[0].page_no)
    except Exception as exc:
        logger.debug("Не удалось извлечь page_no из dl_meta: {}", exc)
        return None


def _page_from_native_chunk(chunk: Any) -> int | None:
    """Пытается извлечь номер страницы из чанка Docling (разные версии API)."""
    try:
        meta = getattr(chunk, "meta", None)
        if meta is None:
            return None
        # Некоторые чанки хранят provenance в meta.doc_items / prov
        doc_items = getattr(meta, "doc_items", None)
        if doc_items:
            first = doc_items[0]
            prov = getattr(first, "prov", None)
            if prov:
                p = prov[0]
                return int(getattr(p, "page_no", None) or getattr(p, "page", None))
    except Exception as exc:
        logger.debug("_page_from_native_chunk: {}", exc)
    return None


def convert_path_to_documents(path: str) -> list[Document]:
    """
    DocumentConverter + HybridChunker → список Haystack Document.
    В meta до attach_file_metadata: chunk_index, page_no (если известна).
    """
    from docling.document_converter import DocumentConverter

    logger.info("DocLing: convert source={}", path)
    dl_doc = DocumentConverter().convert(source=path).document
    chunker = _build_hybrid_chunker()

    out: list[Document] = []
    for idx, ch in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = chunker.contextualize(chunk=ch).strip()
        if not text:
            continue
        page = _page_from_native_chunk(ch)
        if page is None:
            page = -1
        out.append(
            Document(
                id=str(uuid.uuid4()),
                content=text,
                meta={"chunk_index": idx, "page_no": page},
            )
        )
    return out


def attach_file_metadata(
    documents: list[Document],
    *,
    user_id: int,
    filename: str,
    username: str | None,
) -> list[Document]:
    ts = datetime.now(timezone.utc).isoformat()
    uid = str(user_id)
    result: list[Document] = []
    for chunk_index, doc in enumerate(documents):
        prev = doc.meta or {}
        page_no = prev.get("page_no")
        if not isinstance(page_no, int) or page_no < 0:
            page_no = _page_from_dl_meta(prev.get("dl_meta"))
            page_no = int(page_no) if page_no is not None else -1

        flat_meta: dict[str, Any] = {
            "user_id": uid,
            "type": META_TYPE_FILE_CHUNK,
            "filename": filename,
            "chunk_index": int(chunk_index),
            "page_no": int(page_no),
            "username": username or "",
            "timestamp": ts,
        }
        new_id = str(uuid.uuid4())
        result.append(
            replace(
                doc,
                id=new_id,
                meta=flat_meta,
            )
        )
    return result
