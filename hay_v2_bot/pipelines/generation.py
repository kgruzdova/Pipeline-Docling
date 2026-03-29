"""Системные промпты и краткое резюме по загруженному файлу (одно предложение)."""

from __future__ import annotations

import os

from loguru import logger
from openai import OpenAI

from hay_v2_bot.config import CHAT_MODEL, OPENAI_API_BASE_URL


BASE_SYSTEM = """You are a warm, capable personal assistant in Telegram.
Stay coherent with the conversation: use short-term thread and long-term memory when provided.
Long-term memory may include past user messages and excerpts from files the user uploaded; use them when relevant.
Answer in the same language as the user.
Tools: catFactTool (random cat fact from catfact.ninja for questions about cats),
dogImageTool (random dog image URL), dogFactTool (random dog fact about dogs),
docImageAnalyzerTool (random dog photo + vision JSON for Telegram — do not rewrite the tool JSON),
weatherTool (current weather by city name using OpenWeather).
Use tools only when they fit the user's request.
If memory snippets are empty, rely on the current chat only."""


def make_system_prompt(memory_block: str) -> str:
    if memory_block.strip():
        return (
            f"{BASE_SYSTEM}\n\n"
            f"Long-term memory (relevant snippets, cosine-ranked):\n{memory_block}"
        )
    return f"{BASE_SYSTEM}\n\n(Long-term memory: no close matches for this query.)"


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=OPENAI_API_BASE_URL,
    )


def summarize_uploaded_file_one_sentence(combined_text: str) -> str:
    """
    Одно короткое предложение о содержании (русский), по выборке текста из документа.
    """
    sample = (combined_text or "").strip()
    if len(sample) > 14000:
        sample = sample[:14000] + "\n…"
    if not sample:
        return "Содержимое документа пустое или не удалось извлечь текст."

    prompt = (
        "Напиши ровно одно короткое предложение на русском языке: "
        "о чём этот документ и какова его главная тема. Без вступлений и кавычек.\n\n"
        f"Текст (фрагмент):\n{sample}"
    )
    try:
        resp = _openai_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        out = (resp.choices[0].message.content or "").strip()
        one = out.split("\n")[0].strip()
        return one if one else out
    except Exception as exc:
        logger.exception("summarize_uploaded_file_one_sentence: {}", exc)
        return "Не удалось сформировать краткое резюме документа."
