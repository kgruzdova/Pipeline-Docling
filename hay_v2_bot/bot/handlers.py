"""Обработчики Telegram: команды, текст, документы."""

from __future__ import annotations

import os
import tempfile
from collections import deque

import telebot
from haystack.dataclasses import ChatMessage
from loguru import logger

from hay_v2_bot.bot.context import BotRuntime
from hay_v2_bot.components.tools import (
    extract_doc_analyzer_photo_from_messages,
    log_tool_results_summary,
    split_telegram,
    user_message_mentions_cat_facts,
)
from hay_v2_bot.config import (
    PINECONE_NAMESPACE,
    SHORT_TERM_MAX_MESSAGES,
    TELEGRAM_CAPTION_MAX,
    TELEGRAM_MAX_LEN,
)
from hay_v2_bot.pipelines.generation import make_system_prompt, summarize_uploaded_file_one_sentence
from hay_v2_bot.pipelines.ingestion import run_ingestion_pipeline
from hay_v2_bot.pipelines.memory import forget_user_memory, persist_turn, retrieve_memory_context

_short_term: dict[int, deque] = {}
_rt: BotRuntime | None = None


def get_history(user_id: int) -> deque:
    if user_id not in _short_term:
        _short_term[user_id] = deque(maxlen=SHORT_TERM_MAX_MESSAGES)
    return _short_term[user_id]


def register_handlers(bot: telebot.TeleBot, runtime: BotRuntime) -> None:
    global _rt
    _rt = runtime

    @bot.message_handler(commands=["start"])
    def handle_start(message: telebot.types.Message) -> None:
        name = (message.from_user.first_name if message.from_user else None) or "друг"
        bot.send_message(
            message.chat.id,
            f"Привет, {name}!\n\n"
            "Я твой умный персональный помощник (Haystack v2). Я запоминаю наши разговоры, "
            "могу принимать документы (PDF, DOCX и др.), анализировать их через DocLing и "
            "учитывать их при ответах.\n\n"
            "Я умею:\n"
            "• Помнить контекст и фрагменты загруженных файлов\n"
            "• Получать случайные факты о кошках 🐱 и о собаках 🐶\n"
            "• Показывать картинки собак и определять их породы 🖼️\n"
            "• Погода по городу\n\n"
            "Отправь текст или файл.\n\n"
            "Команды:\n"
            "/start — приветствие\n"
            "/clear — очистить краткосрочный буфер диалога\n"
            "/forget — удалить долговременную память в Pinecone (namespace этого бота)",
        )

    @bot.message_handler(commands=["clear"])
    def handle_clear(message: telebot.types.Message) -> None:
        uid = message.from_user.id
        _short_term.pop(uid, None)
        bot.send_message(message.chat.id, "Short-term buffer cleared.")

    @bot.message_handler(commands=["forget"])
    def handle_forget(message: telebot.types.Message) -> None:
        assert _rt is not None
        uid = message.from_user.id
        _short_term.pop(uid, None)
        try:
            n = forget_user_memory(_rt.document_store, uid)
            bot.send_message(
                message.chat.id,
                f"Removed {n} long-term memory records for you in namespace '{PINECONE_NAMESPACE}'.",
            )
        except Exception as exc:
            logger.exception("forget failed: {}", exc)
            bot.send_message(message.chat.id, "Could not erase long-term memory. Try again later.")

    @bot.message_handler(content_types=["document"])
    def handle_document(message: telebot.types.Message) -> None:
        assert _rt is not None
        user_id = message.from_user.id
        uname = (message.from_user.username if message.from_user else None) or ""

        doc = message.document
        if not doc:
            return

        fname = doc.file_name or "uploaded_file"
        logger.info("handle_document: file_name={} mime={} size={}", fname, doc.mime_type, doc.file_size)

        bot.send_chat_action(message.chat.id, "typing")
        bot.send_message(
            message.chat.id,
            "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…",
        )

        file_info = bot.get_file(doc.file_id)
        try:
            data = bot.download_file(file_info.file_path)
        except Exception as exc:
            logger.exception("download_file failed: {}", exc)
            bot.send_message(message.chat.id, "Не удалось скачать файл. Попробуйте ещё раз.")
            return

        suffix = os.path.splitext(fname)[1] or ".bin"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            n, combined_preview = run_ingestion_pipeline(
                local_path=tmp_path,
                user_id=user_id,
                filename=fname,
                username=uname,
                document_embedder=_rt.document_embedder,
                document_store=_rt.document_store,
            )

            if n <= 0:
                bot.send_message(
                    message.chat.id,
                    "Не удалось извлечь текст из файла или формат не поддерживается.",
                )
                return

            sentence = summarize_uploaded_file_one_sentence(combined_preview)

            bot.send_message(
                message.chat.id,
                "Готово. Я изучил этот файл, теперь можем его обсудить.",
            )
            bot.send_message(message.chat.id, sentence)
        except Exception as exc:
            logger.exception("handle_document: {}", exc)
            bot.send_message(
                message.chat.id,
                "Ошибка при обработке файла. Попробуйте другой формат или позже.",
            )
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @bot.message_handler(func=lambda m: m.content_type == "text")
    def handle_text(message: telebot.types.Message) -> None:
        assert _rt is not None
        user_id = message.from_user.id
        user_text = (message.text or "").strip()
        if not user_text:
            return

        bot.send_chat_action(message.chat.id, "typing")

        try:
            _rt.agent.warm_up()

            uname = (message.from_user.username if message.from_user else None) or ""
            if user_message_mentions_cat_facts(user_text):
                logger.info(
                    "handle_text: запрос про кошек; query: {}",
                    user_text[:300] + ("..." if len(user_text) > 300 else ""),
                )

            mem = retrieve_memory_context(
                user_id,
                user_text,
                _rt.text_embedder,
                _rt.retriever,
            )
            system_prompt = make_system_prompt(mem)

            hist = list(get_history(user_id))
            runtime_messages = hist + [ChatMessage.from_user(user_text)]

            logger.info(
                "handle_text: Agent, user_id={}, history_msgs={}",
                user_id,
                len(runtime_messages),
            )
            result = _rt.agent.run(messages=runtime_messages, system_prompt=system_prompt)
            final_messages = result.get("messages") or []
            if not final_messages:
                bot.send_message(message.chat.id, "No response from agent.")
                return
            reply = final_messages[-1].text or ""
            log_tool_results_summary(final_messages)
            photo_url, photo_caption = extract_doc_analyzer_photo_from_messages(final_messages)

            if photo_url and photo_caption is not None:
                cap = photo_caption[:TELEGRAM_CAPTION_MAX]
                try:
                    bot.send_photo(message.chat.id, photo_url, caption=cap)
                except Exception as exc:
                    logger.exception("send_photo failed: {}", exc)
                    bot.send_message(
                        message.chat.id,
                        f"{photo_url}\n\n{photo_caption}",
                    )
                assistant_for_memory = photo_caption if photo_caption.strip() else reply
                if reply.strip() and reply.strip() != photo_caption.strip():
                    for chunk in split_telegram(reply, TELEGRAM_MAX_LEN):
                        bot.send_message(message.chat.id, chunk)
                    assistant_for_memory = f"{photo_caption}\n\n{reply}".strip()
            else:
                for chunk in split_telegram(reply, TELEGRAM_MAX_LEN):
                    bot.send_message(message.chat.id, chunk)
                assistant_for_memory = reply

            get_history(user_id).append(ChatMessage.from_user(user_text))
            get_history(user_id).append(ChatMessage.from_assistant(assistant_for_memory))

            try:
                persist_turn(
                    user_id,
                    user_text,
                    _rt.document_embedder,
                    _rt.document_store,
                    username=uname,
                )
            except Exception as persist_exc:
                logger.warning("Pinecone write failed (reply still sent): {}", persist_exc)
        except Exception as exc:
            logger.exception("handler error: {}", exc)
            bot.send_message(message.chat.id, "Something went wrong. Please try again.")
