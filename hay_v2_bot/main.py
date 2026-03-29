"""
Точка входа Telegram-бота v2 (Haystack + DocLing + Pinecone).

Запуск из корня репозитория:
  python hay_v2_bot/main.py
или:
  python -m hay_v2_bot.main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта пакета hay_v2_bot
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import telebot
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from loguru import logger

from hay_v2_bot.bot.context import BotRuntime
from hay_v2_bot.bot.handlers import register_handlers
from hay_v2_bot.components.embedders import build_embedders
from hay_v2_bot.components.logging_setup import init_logging
from hay_v2_bot.components.pinecone_store import build_document_store
from hay_v2_bot.config import CHAT_MODEL, MEMORY_TOP_K, PINECONE_DIMENSION, PINECONE_INDEX, PINECONE_NAMESPACE
from hay_v2_bot.pipelines.agent_factory import build_agent


def main() -> None:
    init_logging()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env")

    document_store = build_document_store()
    text_embedder, document_embedder = build_embedders()
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=MEMORY_TOP_K)
    agent = build_agent()

    agent.warm_up()

    runtime = BotRuntime(
        document_store=document_store,
        text_embedder=text_embedder,
        document_embedder=document_embedder,
        retriever=retriever,
        agent=agent,
    )

    bot = telebot.TeleBot(token, parse_mode=None)
    register_handlers(bot, runtime)

    logger.info(
        "Haystack Telegram bot v2 | index={} namespace={} dim={} model={}",
        PINECONE_INDEX,
        PINECONE_NAMESPACE,
        PINECONE_DIMENSION,
        CHAT_MODEL,
    )
    bot.infinity_polling(timeout=30, long_polling_timeout=20)


if __name__ == "__main__":
    main()
