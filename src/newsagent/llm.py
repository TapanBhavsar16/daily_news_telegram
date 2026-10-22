from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
import httpx

from newsagent.config import AgentConfig, Config, get_settings
from newsagent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str


@dataclass
class ChatResponse:
    content: Optional[str]
    tool_calls: list[ToolCall]
    usage: Optional[dict[str, int]] = None
    model: Optional[str] = None


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[ToolSchema]] = None,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        pass

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        pass


class NativeToolClient(LLMClient):
    def __init__(self, model: str, fallback_models: list[str] = None):
        self.model = model
        self.fallback_models = fallback_models or []
        self._check_support()

    def _check_support(self) -> None:
        try:
            self._supports_tools = litellm.supports_function_calling(model=self.model)
        except Exception:
            self._supports_tools = False

    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[ToolSchema]] = None,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        formatted_messages = messages.copy()
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})

        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        models_to_try = [self.model] + self.fallback_models
        last_error = None

        for model in models_to_try:
            try:
                response = self._call_with_retry(
                    model=model,
                    messages=formatted_messages,
                    tools=tool_schemas,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                logger.warning("Model failed, trying fallback", model=model, error=str(e))
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, litellm.RateLimitError)),
    )
    def _call_with_retry(self, model: str, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]], temperature: float, max_tokens: Optional[int]) -> Any:
        return litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _parse_response(self, response: Any) -> ChatResponse:
        message = response.choices[0].message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        id=tc.id,
                    )
                )
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
            model=response.model,
        )


class JSONActionClient(LLMClient):
    def __init__(self, model: str, fallback_models: list[str] = None):
        self.model = model
        self.fallback_models = fallback_models or []

    def supports_tool_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[ToolSchema]] = None,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        formatted_messages = messages.copy()

        tool_descriptions = ""
        if tools:
            tool_descriptions = "\n\nAvailable tools:\n"
            for t in tools:
                tool_descriptions += f"- {t.name}: {t.description}\n"
                tool_descriptions += f"  Parameters: {json.dumps(t.parameters)}\n"

        json_system = (
            (system or "")
            + tool_descriptions
            + """

You must respond with exactly one JSON object per turn. Format:
{
  "thought": "your reasoning",
  "tool": "tool_name",
  "args": {"param": "value"}
}

If you have the final answer, use tool "finish" with args {"answer": "...", "confidence": 0.0-1.0, "open_gaps": []}.
Only call one tool per response. No extra text outside the JSON.
"""
        )

        formatted_messages.insert(0, {"role": "system", "content": json_system})

        models_to_try = [self.model] + self.fallback_models
        last_error = None

        for model in models_to_try:
            try:
                response = self._call_with_retry(
                    model=model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._parse_json_response(response)
            except Exception as e:
                last_error = e
                logger.warning("Model failed, trying fallback", model=model, error=str(e))
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, litellm.RateLimitError)),
    )
    def _call_with_retry(self, model: str, messages: list[dict[str, Any]], temperature: float, max_tokens: Optional[int]) -> Any:
        return litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _parse_json_response(self, response: Any) -> ChatResponse:
        import re
        message = response.choices[0].message
        content = message.content or ""

        json_text = content.strip()

        def try_parse(text):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                for start in range(len(text)):
                    for end in range(start + 1, len(text) + 1):
                        if text[start] == '{' and text[end-1] == '}':
                            try:
                                return json.loads(text[start:end])
                            except json.JSONDecodeError:
                                pass
            return None

        parsed = try_parse(json_text)

        if parsed is None:
            return ChatResponse(
                content=content,
                tool_calls=[],
                usage=(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
                model=response.model,
            )

        tool_name = parsed.get("tool")
        if not tool_name:
            return ChatResponse(
                content=content,
                tool_calls=[],
                usage=(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
                model=response.model,
            )

        return ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    name=tool_name,
                    arguments=parsed.get("args", {}),
                    id=f"call_{hash(content)}",
                )
            ],
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
            model=response.model,
        )


def create_llm_client(config: AgentConfig, llm_config: Config) -> LLMClient:
    settings = get_settings()
    api_key = settings.NVIDIA_API_KEY or settings.OPENAI_API_KEY
    if api_key:
        os.environ["NVIDIA_NIM_API_KEY"] = api_key
        os.environ["NVIDIA_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    tool_mode = config.tool_mode

    if tool_mode == "native":
        return NativeToolClient(
            model=llm_config.llm.model,
            fallback_models=llm_config.llm.fallbacks,
        )
    elif tool_mode == "json":
        return JSONActionClient(
            model=llm_config.llm.model,
            fallback_models=llm_config.llm.fallbacks,
        )
    else:
        native = NativeToolClient(
            model=llm_config.llm.model,
            fallback_models=llm_config.llm.fallbacks,
        )
        if native.supports_tool_calling():
            return native
        return JSONActionClient(
            model=llm_config.llm.model,
            fallback_models=llm_config.llm.fallbacks,
        )


def create_small_llm_client(config: AgentConfig, llm_config: Config) -> LLMClient:
    settings = get_settings()
    api_key = settings.NVIDIA_API_KEY or settings.OPENAI_API_KEY
    if api_key:
        os.environ["NVIDIA_NIM_API_KEY"] = api_key
        os.environ["NVIDIA_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    tool_mode = config.tool_mode

    if tool_mode == "native":
        return NativeToolClient(
            model=llm_config.llm.small_model,
            fallback_models=llm_config.llm.fallbacks,
        )
    elif tool_mode == "json":
        return JSONActionClient(
            model=llm_config.llm.small_model,
            fallback_models=llm_config.llm.fallbacks,
        )
    else:
        native = NativeToolClient(
            model=llm_config.llm.small_model,
            fallback_models=llm_config.llm.fallbacks,
        )
        if native.supports_tool_calling():
            return native
        return JSONActionClient(
            model=llm_config.llm.small_model,
            fallback_models=llm_config.llm.fallbacks,
        )