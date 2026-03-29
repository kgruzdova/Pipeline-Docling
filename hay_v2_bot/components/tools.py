"""Инструменты Haystack Agent (паритет с hay-telegram-bot)."""

from __future__ import annotations

import json
import os
import re

import requests
from haystack.dataclasses import ChatMessage, TextContent, ToolCallResult
from loguru import logger
from openai import OpenAI

from hay_v2_bot.config import (
    CAT_FACT_API,
    CHAT_MODEL,
    DOG_CEO_RANDOM_IMAGE,
    DOG_FACT_API,
    OPENAI_API_BASE_URL,
    OPENWEATHER_API_KEY,
    OPENWEATHER_CURRENT_BY_CITY,
    TOOL_NAME_CAT_FACT,
    TOOL_NAME_DOC_IMAGE_ANALYZER,
    TOOL_NAME_WEATHER,
)


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=OPENAI_API_BASE_URL,
    )


def cat_fact_tool() -> str:
    """Случайный факт о кошках (catfact.ninja)."""
    logger.info("CatFactTool.run: запрос к API {}", CAT_FACT_API)
    try:
        r = requests.get(CAT_FACT_API, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        fact = (data.get("fact") or "").strip()
    except Exception as exc:
        logger.exception("CatFactTool.run: ошибка API: {}", exc)
        return "Не удалось получить факт о кошках (ошибка сети или API)."
    if not fact:
        return "Факт о кошках не пришёл."
    logger.info("CatFactTool.run: факт получен, длина: {}", len(fact))
    return f"Факт о кошках: {fact}"


def dog_image_tool() -> str:
    """Случайный URL изображения собаки (dog.ceo)."""
    logger.info("dogImageTool.run: запрос {}", DOG_CEO_RANDOM_IMAGE)
    r = requests.get(DOG_CEO_RANDOM_IMAGE, timeout=25)
    r.raise_for_status()
    image_url = (r.json() or {}).get("message", "").strip()
    if not image_url:
        return "Не удалось получить URL изображения собаки."
    return f"Случайное изображение собаки: {image_url}"


def dog_fact_tool() -> str:
    """Случайный факт о собаках (kinduff)."""
    logger.info("DogFactTool.run: запрос к API {}", DOG_FACT_API)
    try:
        r = requests.get(DOG_FACT_API, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        facts = data.get("facts") or []
        fact = (facts[0] if facts else "").strip()
    except Exception as exc:
        logger.exception("DogFactTool.run: ошибка API: {}", exc)
        return "Не удалось получить факт о собаках (ошибка сети или API)."
    if not fact:
        return "Факт о собаках не пришёл."
    return f"Факт о собаках: {fact}"


def weather_openweather_tool(city_query: str) -> str:
    """Текущая погода по названию города (OpenWeather)."""
    place = (city_query or "").strip()
    if not place:
        return "Укажите название города для запроса погоды."

    if not OPENWEATHER_API_KEY:
        logger.error("{}.run: OPENWEATHER_API_KEY не задан", TOOL_NAME_WEATHER)
        return "Сервис погоды недоступен: не настроен `OPENWEATHER_API_KEY`."

    params = {
        "q": place,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ru",
    }
    try:
        r = requests.get(OPENWEATHER_CURRENT_BY_CITY, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as exc:
        logger.exception("{}.run: ошибка OpenWeather API: {}", TOOL_NAME_WEATHER, exc)
        return "Не удалось получить погоду (ошибка сети/сервиса OpenWeather)."

    if data.get("cod") not in (200, "200"):
        msg = data.get("message") or "unknown error"
        return f"Не удалось получить погоду для «{place}»: {msg}."

    main = data.get("main") or {}
    weather_arr = data.get("weather") or []
    wind = data.get("wind") or {}
    temp = main.get("temp")
    feels = main.get("feels_like")
    humidity = main.get("humidity")
    wind_speed = wind.get("speed")
    wind_deg = wind.get("deg")
    weather_desc = ""
    if weather_arr:
        weather_desc = (weather_arr[0].get("description") or "").strip()
    city_name = data.get("name") or place
    country = (data.get("sys") or {}).get("country") or ""
    location_line = f"{city_name}" + (f" ({country})" if country else "")

    parts = [f"Погода в {location_line}:"]
    if weather_desc:
        parts.append(f"{weather_desc}.")
    if temp is not None:
        parts.append(f"Температура: {temp} °C.")
    if feels is not None:
        parts.append(f"Ощущается как: {feels} °C.")
    if humidity is not None:
        parts.append(f"Влажность: {humidity}%.")
    if wind_speed is not None:
        wind_str = f"{wind_speed} м/с"
        if wind_deg is not None:
            wind_str += f" (на {wind_deg}°)"
        parts.append(f"Ветер: {wind_str}.")
    parts.append("(Источник: OpenWeather, бесплатные данные тарифа.)")
    return " ".join(parts)


def doc_image_analyzer_tool() -> str:
    """
    Случайное фото собаки + vision; возвращает JSON для Telegram (photo + caption).
    """
    logger.info("{}.run: загрузка URL изображения", TOOL_NAME_DOC_IMAGE_ANALYZER)
    r = requests.get(DOG_CEO_RANDOM_IMAGE, timeout=25)
    r.raise_for_status()
    image_url = (r.json() or {}).get("message", "").strip()
    if not image_url:
        return json.dumps(
            {"error": "no_image_url", "message": "Could not get a dog image URL."},
            ensure_ascii=False,
        )

    client = _openai_client()
    prompt = (
        "Look at the dog in this image. Name the most likely breed (or mix). "
        "Give 3–6 sentences: brief breed traits, geographic/historical origin, "
        "how the breed developed. Reply in the same language the Telegram user likely uses "
        "(if unclear, use Russian)."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=600,
    )
    desc = (resp.choices[0].message.content or "").strip()
    payload = {
        "_telegram": "send_photo",
        "photo_url": image_url,
        "caption": desc,
    }
    return json.dumps(payload, ensure_ascii=False)


def _tool_call_result_as_str(tcr: ToolCallResult) -> str:
    res = tcr.result
    if isinstance(res, str):
        return res
    if isinstance(res, list):
        parts: list[str] = []
        for part in res:
            if isinstance(part, TextContent):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(res)


def extract_doc_analyzer_photo_from_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, str | None]:
    for msg in messages:
        for tcr in msg.tool_call_results:
            if tcr.origin.tool_name != TOOL_NAME_DOC_IMAGE_ANALYZER:
                continue
            raw = _tool_call_result_as_str(tcr).strip()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{[\s\S]*\"_telegram\"[\s\S]*\}", raw)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
            if obj.get("_telegram") != "send_photo":
                continue
            url, cap = obj.get("photo_url"), obj.get("caption")
            if url and cap is not None:
                return str(url), str(cap)
    return None, None


def user_message_mentions_cat_facts(text: str) -> bool:
    t = (text or "").lower()
    needles = (
        "кош",
        "кот",
        "котя",
        "коты",
        "коте",
        "котов",
        "кошк",
        "cat fact",
        "facts about cats",
        "факт о кош",
        "факт про кош",
        "факты о кош",
        "кошач",
    )
    return any(n in t for n in needles)


def log_tool_results_summary(messages: list[ChatMessage]) -> None:
    for msg in messages:
        for tcr in msg.tool_call_results:
            name = tcr.origin.tool_name
            raw = _tool_call_result_as_str(tcr)
            logger.info(
                "handle_text: результат инструмента {}, длина вывода: {}",
                name,
                len(raw),
            )
            if name == TOOL_NAME_CAT_FACT:
                logger.info(
                    "handle_text: catFactTool — превью: {}",
                    raw[:120] + ("..." if len(raw) > 120 else ""),
                )


def split_telegram(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        parts.append(rest[:limit])
        rest = rest[limit:]
    return parts
