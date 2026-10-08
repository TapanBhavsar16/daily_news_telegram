from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from newsagent.config import AgentConfig, Config, load_config
from newsagent.llm import (
    ChatResponse,
    JSONActionClient,
    LLMClient,
    NativeToolClient,
    ToolCall,
    ToolSchema,
    create_llm_client,
)
from newsagent.logging import setup_logging


@dataclass
class FakeToolCall:
    function: Any
    id: str


class FakeChoice:
    def __init__(self, message: Any):
        self.message = message


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class FakeResponse:
    def __init__(self, choices: list, model: str = "test-model"):
        self.choices = choices
        self.model = model
        self.usage = FakeUsage()


class FakeMessage:
    def __init__(self, content: str = None, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls


def make_fake_native_response(tool_name: str = None, tool_args: dict = None, content: str = None):
    tool_calls = []
    if tool_name:
        mock_func = MagicMock()
        mock_func.name = tool_name
        mock_func.arguments = json.dumps(tool_args or {})
        tool_calls.append(
            FakeToolCall(
                function=mock_func,
                id="call_123",
            )
        )
    message = FakeMessage(content=content, tool_calls=tool_calls if tool_calls else None)
    return FakeResponse(choices=[FakeChoice(message)])


def make_fake_json_response(json_str: str):
    message = FakeMessage(content=json_str, tool_calls=None)
    return FakeResponse(choices=[FakeChoice(message)])


class MockNativeClient(LLMClient):
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self._supports_tools = True

    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def chat(self, messages, tools=None, system=None, temperature=0.0, max_tokens=None):
        if self.call_count >= len(self.responses):
            return ChatResponse(content="Done", tool_calls=[])
        resp = self.responses[self.call_count]
        self.call_count += 1
        return self._parse_response(resp)

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
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
        )


class MockJSONClient(LLMClient):
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self._supports_tools = True

    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def chat(self, messages, tools=None, system=None, temperature=0.0, max_tokens=None):
        if self.call_count >= len(self.responses):
            return ChatResponse(content="Done", tool_calls=[])
        resp = self.responses[self.call_count]
        self.call_count += 1
        return self._parse_json_response(resp)

    def _parse_json_response(self, response: Any) -> ChatResponse:
        message = response.choices[0].message
        content = message.content or ""

        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError:
            return ChatResponse(
                content=content,
                tool_calls=[],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                model=response.model,
            )

        tool_name = parsed.get("tool")
        if not tool_name:
            return ChatResponse(
                content=content,
                tool_calls=[],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
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
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
        )


TOOL_SCHEMAS = [
    ToolSchema(
        name="search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolSchema(
        name="finish",
        description="Finish the task",
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "open_gaps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence"],
        },
    ),
]


def run_toy_loop(client: LLMClient, goal: str, max_steps: int = 5) -> dict:
    messages = [
        {"role": "system", "content": "You are a test agent. Use tools to answer."},
        {"role": "user", "content": goal},
    ]

    for step in range(max_steps):
        resp = client.chat(messages, tools=TOOL_SCHEMAS)
        messages.append({"role": "assistant", "content": resp.content or ""})

        if not resp.tool_calls:
            return {"result": resp.content, "steps": step + 1, "reason": "text_answer"}

        for call in resp.tool_calls:
            if call.name == "finish":
                return {
                    "result": call.arguments.get("answer"),
                    "steps": step + 1,
                    "reason": "finish",
                    "confidence": call.arguments.get("confidence"),
                }
            elif call.name == "search":
                tool_result = {"results": [{"title": "Test", "url": "http://test.com", "snippet": "Test result"}]}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result),
                    }
                )

    return {"result": "Max steps reached", "steps": max_steps, "reason": "max_steps"}


def test_native_tool_mode():
    responses = [
        make_fake_native_response("search", {"query": "test query", "num_results": 3}),
        make_fake_native_response("finish", {"answer": "Test answer", "confidence": 0.9, "open_gaps": []}),
    ]
    client = MockNativeClient(responses)
    result = run_toy_loop(client, "Test goal", max_steps=5)

    assert result["reason"] == "finish"
    assert result["result"] == "Test answer"
    assert result["confidence"] == 0.9
    assert client.call_count == 2


def test_json_tool_mode():
    responses = [
        make_fake_json_response(
            json.dumps(
                {
                    "thought": "Need to search",
                    "tool": "search",
                    "args": {"query": "test query", "num_results": 3},
                }
            )
        ),
        make_fake_json_response(
            json.dumps(
                {
                    "thought": "Have answer",
                    "tool": "finish",
                    "args": {"answer": "Test answer", "confidence": 0.9, "open_gaps": []},
                }
            )
        ),
    ]
    client = MockJSONClient(responses)
    result = run_toy_loop(client, "Test goal", max_steps=5)

    assert result["reason"] == "finish"
    assert result["result"] == "Test answer"
    assert result["confidence"] == 0.9
    assert client.call_count == 2


def test_both_modes_produce_same_result():
    native_responses = [
        make_fake_native_response("search", {"query": "test", "num_results": 3}),
        make_fake_native_response("finish", {"answer": "Same answer", "confidence": 0.8, "open_gaps": []}),
    ]
    json_responses = [
        make_fake_json_response(
            json.dumps(
                {"thought": "Search", "tool": "search", "args": {"query": "test", "num_results": 3}}
            )
        ),
        make_fake_json_response(
            json.dumps(
                {"thought": "Done", "tool": "finish", "args": {"answer": "Same answer", "confidence": 0.8, "open_gaps": []}}
            )
        ),
    ]

    native_client = MockNativeClient(native_responses)
    json_client = MockJSONClient(json_responses)

    native_result = run_toy_loop(native_client, "Test goal")
    json_result = run_toy_loop(json_client, "Test goal")

    assert native_result["result"] == json_result["result"]
    assert native_result["confidence"] == json_result["confidence"]
    assert native_result["reason"] == json_result["reason"]


def test_json_mode_parse_error_recovery():
    # In the mock, a parse error returns content with no tool_calls, which run_toy_loop treats as text_answer
    # The real JSONActionClient would handle this differently (send error back, retry)
    # This test verifies the mock behavior
    responses = [
        make_fake_json_response("not valid json"),
        make_fake_json_response(
            json.dumps(
                {"thought": "Retry", "tool": "finish", "args": {"answer": "Recovered", "confidence": 0.7, "open_gaps": []}}
            )
        ),
    ]
    client = MockJSONClient(responses)
    result = run_toy_loop(client, "Test goal", max_steps=5)

    # Mock returns text_answer on parse error (first response has no tool_calls)
    assert result["reason"] == "text_answer"
    assert "not valid json" in result["result"]


if __name__ == "__main__":
    setup_logging(Config().logging)
    test_native_tool_mode()
    print("✓ Native tool mode test passed")
    test_json_tool_mode()
    print("✓ JSON tool mode test passed")
    test_both_modes_produce_same_result()
    print("✓ Both modes produce same result test passed")
    test_json_mode_parse_error_recovery()
    print("✓ JSON mode parse error recovery test passed")
    print("\nAll toy tool loop tests passed!")