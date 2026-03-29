"""Сборка Haystack Agent с инструментами (паритет с v1)."""

from __future__ import annotations

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import create_tool_from_function
from haystack.utils import Secret

from hay_v2_bot.components import tools
from hay_v2_bot.config import CHAT_MODEL, OPENAI_API_BASE_URL, TOOL_NAME_CAT_FACT, TOOL_NAME_DOC_IMAGE_ANALYZER, TOOL_NAME_WEATHER
from hay_v2_bot.pipelines.generation import BASE_SYSTEM


def build_agent() -> Agent:
    chat_gen = OpenAIChatGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=CHAT_MODEL,
        api_base_url=OPENAI_API_BASE_URL,
    )

    agent_tools = [
        create_tool_from_function(tools.cat_fact_tool, name=TOOL_NAME_CAT_FACT),
        create_tool_from_function(tools.dog_image_tool, name="dogImageTool"),
        create_tool_from_function(tools.dog_fact_tool, name="dogFactTool"),
        create_tool_from_function(tools.doc_image_analyzer_tool, name=TOOL_NAME_DOC_IMAGE_ANALYZER),
        create_tool_from_function(tools.weather_openweather_tool, name=TOOL_NAME_WEATHER),
    ]

    return Agent(
        chat_generator=chat_gen,
        tools=agent_tools,
        system_prompt=BASE_SYSTEM,
        exit_conditions=["text"],
        max_agent_steps=20,
        raise_on_tool_invocation_failure=False,
    )
