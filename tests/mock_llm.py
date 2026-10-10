from __future__ import annotations

import json
from typing import Any, Optional

from newsagent.llm import LLMClient, ChatResponse, ToolCall, ToolSchema


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[dict[str, Any]] = None):
        self.responses = responses or []
        self.call_count = 0

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
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = {"tool": "finish", "args": {"answer": "Mock answer", "confidence": 0.8, "open_gaps": []}}

        self.call_count += 1

        tool_name = resp.get("tool", "finish")
        return ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    name=tool_name,
                    arguments=resp.get("args", {}),
                    id=f"call_{self.call_count}",
                )
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock",
        )