"""Переменные окружения и константы бота v2."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

SHORT_TERM_MAX_MESSAGES: int = 24
MEMORY_TOP_K: int = int(os.getenv("HAYSTACK_V2_MEMORY_TOP_K", "12"))
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# Эмбеддинги только через OpenAI API (прокси), локальных моделей нет.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
PINECONE_INDEX: str = os.getenv("PINECONE_INDEX_NAME", "default")
PINECONE_NAMESPACE: str = os.getenv("HAYSTACK_PINECONE_NAMESPACE", "haystack_memory")
PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1536"))
TELEGRAM_MAX_LEN: int = 4096
TELEGRAM_CAPTION_MAX: int = 1024


def _require_openai_api_base_url() -> str:
    """
    Базовый URL OpenAI-совместимого API (обязателен: весь трафик к LLM/эмбеддингам через прокси).
    Пример: https://api.proxyapi.ru/openai/v1
    """
    base = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "OPENAI_BASE_URL не задан в окружении. Укажите прокси, например: "
            "OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1"
        )
    return base


# Единый base URL для Haystack OpenAI-компонентов и официального openai.AsyncClient/OpenAI.
OPENAI_API_BASE_URL: str = _require_openai_api_base_url()

DOG_CEO_RANDOM_IMAGE = "https://dog.ceo/api/breeds/image/random"
DOG_FACT_API = "https://dog-api.kinduff.com/api/facts?number=1"
CAT_FACT_API = "https://catfact.ninja/fact"

TOOL_NAME_DOC_IMAGE_ANALYZER = "docImageAnalyzerTool"
TOOL_NAME_CAT_FACT = "catFactTool"
TOOL_NAME_WEATHER = "weatherTool"

OPENWEATHER_API_BASE_URL = os.getenv("OPENWEATHER_API_BASE_URL", "https://api.openweathermap.org/data/2.5")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_CURRENT_BY_CITY = f"{OPENWEATHER_API_BASE_URL}/weather"

# Макс. токенов на чанк DocLing (счёт через tiktoken, совместимый с text-embedding-3-small).
DOCLING_CHUNK_MAX_TOKENS: int = int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "2048"))

META_TYPE_USER_MESSAGE = "user_message"
META_TYPE_FILE_CHUNK = "file_chunk"
