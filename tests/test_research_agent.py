from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newsagent.agents import ResearchAgent, ResearchResult
from newsagent.config import Config
from tests.mock_llm import MockLLMClient


class MockBudget:
    def __init__(self):
        self.max_steps = 10
        self.max_fetches = 5
        self.max_search_calls = 5
        self.max_tokens_total = 10000
        self.wall_clock_seconds = 60
        self.stall_steps = 3
        self.steps_used = 0
        self.fetches_used = 0
        self.search_calls_used = 0
        self.tokens_used = 0
        self.start_time = 0
        self.last_fact_step = 0
        self._exhausted = False
        self._stalled = False

    def exhausted(self):
        return self._exhausted

    def stalled(self, stall_steps=3):
        return self._stalled

    def record_step(self):
        self.steps_used += 1

    def record_fetch(self):
        self.fetches_used += 1

    def record_search(self):
        self.search_calls_used += 1

    def record_tokens(self, tokens):
        self.tokens_used += tokens

    def record_fact(self):
        self.last_fact_step = self.steps_used

    def remaining_summary(self):
        return f"Steps: {self.steps_used}/{self.max_steps}"

    def get_budget_status(self):
        return {
            "steps_used": self.steps_used,
            "steps_remaining": self.max_steps - self.steps_used,
            "fetches_used": self.fetches_used,
            "fetches_remaining": self.max_fetches - self.fetches_used,
            "searches_used": self.search_calls_used,
            "searches_remaining": self.max_search_calls - self.search_calls_used,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.max_tokens_total - self.tokens_used,
            "time_elapsed": 1.0,
            "time_remaining": 59.0,
            "stalled": self._stalled,
            "visited_urls": 0,
            "seen_queries": 0,
        }


@pytest.fixture
def mock_config():
    return Config()


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.get_tool_schemas.return_value = [
        {"function": {"name": "search", "description": "Search", "parameters": {}}},
        {"function": {"name": "fetch", "description": "Fetch", "parameters": {}}},
        {"function": {"name": "scratchpad", "description": "Scratchpad", "parameters": {}}},
        {"function": {"name": "finish", "description": "Finish", "parameters": {}}},
    ]

    async def mock_guarded_execute(name, args, scratchpad=None):
        if name == "search":
            return json.dumps({"results": [{"title": "Test", "url": "http://test.com", "snippet": "Test"}], "query": args.get("query")})
        elif name == "fetch":
            return json.dumps({"title": "Test", "url": "http://test.com", "text": "Test content", "outbound_links": []})
        elif name == "scratchpad":
            return json.dumps({"status": "written", "key": args.get("key")})
        elif name == "finish":
            return json.dumps({"status": "finish", "args": args})
        return json.dumps({"error": "Unknown tool"})

    registry.guarded_execute = mock_guarded_execute
    registry.budget = MockBudget()
    return registry


async def test_research_agent_basic(mock_config, mock_tool_registry):
    mock_responses = [
        {"tool": "scratchpad", "args": {"action": "write", "key": "plan", "content": "Test plan"}},
        {"tool": "scratchpad", "args": {"action": "write", "key": "checklist", "content": "Test checklist"}},
        {"tool": "search", "args": {"query": "Python latest version"}},
        {"tool": "fetch", "args": {"url": "http://test.com"}},
        {"tool": "scratchpad", "args": {"action": "write", "key": "facts_1", "content": "Python 3.12 is latest", "source_url": "http://test.com"}},
        {"tool": "finish", "args": {"answer": "Python 3.12 is the latest version", "confidence": 0.9, "open_gaps": []}},
    ]

    mock_llm = MockLLMClient(mock_responses)

    agent = ResearchAgent(
        config=mock_config,
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
    )

    result = await agent.research("What is the latest version of Python?")

    assert isinstance(result, ResearchResult)
    assert result.answer == "Python 3.12 is the latest version"
    assert result.confidence == 0.9
    assert result.steps_used > 0


async def test_research_agent_budget_exhausted(mock_config, mock_tool_registry):
    mock_tool_registry.budget._exhausted = True
    mock_tool_registry.budget.steps_used = 10

    mock_responses = [
        {"tool": "scratchpad", "args": {"action": "write", "key": "plan", "content": "Test plan"}},
    ]

    mock_llm = MockLLMClient(mock_responses)

    agent = ResearchAgent(
        config=mock_config,
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
    )

    result = await agent.research("What is the latest version of Python?")

    assert isinstance(result, ResearchResult)
    assert result.steps_used >= 1


async def main():
    config = Config()
    
    class MockRegistry:
        def __init__(self):
            self.get_tool_schemas = lambda: [
                {"function": {"name": "search", "description": "Search", "parameters": {}}},
                {"function": {"name": "fetch", "description": "Fetch", "parameters": {}}},
                {"function": {"name": "scratchpad", "description": "Scratchpad", "parameters": {}}},
                {"function": {"name": "finish", "description": "Finish", "parameters": {}}},
            ]
            self.budget = MockBudget()
        
        async def guarded_execute(self, name, args, scratchpad=None):
            if name == "search":
                return json.dumps({"results": [{"title": "Test", "url": "http://test.com", "snippet": "Test"}], "query": args.get("query")})
            elif name == "fetch":
                return json.dumps({"title": "Test", "url": "http://test.com", "text": "Test content", "outbound_links": []})
            elif name == "scratchpad":
                return json.dumps({"status": "written", "key": args.get("key")})
            elif name == "finish":
                return json.dumps({"status": "finish", "args": args})
            return json.dumps({"error": "Unknown tool"})
    
    registry = MockRegistry()

    mock_responses = [
        {"tool": "scratchpad", "args": {"action": "write", "key": "plan", "content": "Test plan"}},
        {"tool": "scratchpad", "args": {"action": "write", "key": "checklist", "content": "Test checklist"}},
        {"tool": "search", "args": {"query": "Python latest version"}},
        {"tool": "fetch", "args": {"url": "http://test.com"}},
        {"tool": "scratchpad", "args": {"action": "write", "key": "facts_1", "content": "Python 3.12 is latest", "source_url": "http://test.com"}},
        {"tool": "finish", "args": {"answer": "Python 3.12 is the latest version", "confidence": 0.9, "open_gaps": []}},
    ]

    mock_llm = MockLLMClient(mock_responses)

    agent = ResearchAgent(
        config=config,
        llm_client=mock_llm,
        tool_registry=registry,
    )

    result = await agent.research("What is the latest version of Python?")

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Steps: {result.steps_used}")
    print(f"Sources: {result.sources}")
    print("Test passed!")


if __name__ == "__main__":
    asyncio.run(main())