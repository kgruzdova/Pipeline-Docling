"""
Логическая цепочка «generation» для текстовых сообщений (аналог RAG + чат).

1. ``retrieve_memory_context`` — Pinecone: эмбеддинг запроса + ``PineconeEmbeddingRetriever``
   с фильтром ``user_id`` (сообщения пользователя и чанки загруженных файлов).
2. ``make_system_prompt`` — сборка системного промпта с блоком памяти.
3. ``Agent.run`` — ``OpenAIChatGenerator`` и инструменты (как в ``hay/hay-telegram-bot.py``).

Отдельная декларативная ``haystack.Pipeline`` с ``PromptBuilder`` здесь не используется,
чтобы сохранить полный паритет с первой версией бота (агент с tool-calling).
"""

from hay_v2_bot.pipelines.generation import make_system_prompt
from hay_v2_bot.pipelines.memory import retrieve_memory_context

__all__ = ["make_system_prompt", "retrieve_memory_context"]
