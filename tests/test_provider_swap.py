from __future__ import annotations

import json
import os

import pytest

from newsagent.config import Config, AgentConfig
from newsagent.llm import create_llm_client, NativeToolClient, JSONActionClient, create_small_llm_client
from newsagent.agents import ResearchAgent, ResearchResult
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
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.llm_calls = 0
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

    def record_token_usage(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens
        self.tokens_used += total_tokens
        self.llm_calls += 1

    def record_fact(self):
        self.last_fact_step = self.steps_used

    def remaining_summary(self):
        return f"Steps: {self.steps_used}/{self.max_steps}"

    def cost_summary(self):
        return {
            "total_tokens": self.tokens_used,
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "llm_calls": self.llm_calls,
            "steps_used": self.steps_used,
            "fetches_used": self.fetches_used,
            "searches_used": self.search_calls_used,
            "wall_time_seconds": 1.0,
        }

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


class TestProviderSwap:
    """Test that the system works with different LLM providers and configurations."""

    def test_native_tool_client_creation(self):
        """Test NativeToolClient can be created with different models."""
        client = NativeToolClient(model="gpt-4", fallback_models=["gpt-3.5-turbo"])
        assert client.model == "gpt-4"
        assert client.fallback_models == ["gpt-3.5-turbo"]

    def test_json_action_client_creation(self):
        """Test JSONActionClient can be created with different models."""
        client = JSONActionClient(model="claude-3-sonnet", fallback_models=["claude-3-haiku"])
        assert client.model == "claude-3-sonnet"
        assert client.fallback_models == ["claude-3-haiku"]

    def test_create_llm_client_native_mode(self):
        """Test create_llm_client with native tool mode."""
        config = Config()
        config.agent.tool_mode = "native"
        
        client = create_llm_client(config.agent, config)
        assert isinstance(client, NativeToolClient)

    def test_create_llm_client_json_mode(self):
        """Test create_llm_client with JSON tool mode."""
        config = Config()
        config.agent.tool_mode = "json"
        
        client = create_llm_client(config.agent, config)
        assert isinstance(client, JSONActionClient)

    def test_create_llm_client_auto_mode(self):
        """Test create_llm_client with auto tool mode."""
        config = Config()
        config.agent.tool_mode = "auto"
        
        client = create_llm_client(config.agent, config)
        # Should be either NativeToolClient or JSONActionClient
        assert isinstance(client, (NativeToolClient, JSONActionClient))

    def test_create_small_llm_client(self):
        """Test create_small_llm_client uses small_model."""
        config = Config()
        config.llm.small_model = "gpt-3.5-turbo"
        
        client = create_small_llm_client(config.agent, config)
        # The small model should be used
        assert client.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_research_with_different_tool_modes(self):
        """Test research agent works with both native and JSON tool modes."""
        mock_responses = [
            {"tool": "scratchpad", "args": {"action": "write", "key": "plan", "content": "Test plan"}},
            {"tool": "scratchpad", "args": {"action": "write", "key": "checklist", "content": "Test checklist"}},
            {"tool": "search", "args": {"query": "test query"}},
            {"tool": "fetch", "args": {"url": "http://test.com"}},
            {"tool": "scratchpad", "args": {"action": "write", "key": "facts_1", "content": "Test fact", "source_url": "http://test.com"}},
            {"tool": "finish", "args": {"answer": "Test answer", "confidence": 0.9, "open_gaps": []}},
        ]
        
        for tool_mode in ["native", "json"]:
            config = Config()
            config.agent.tool_mode = tool_mode
            config.agent.max_steps = 10
            
            mock_llm = MockLLMClient(mock_responses)
            registry = MockRegistry()
            
            agent = ResearchAgent(
                config=config,
                llm_client=mock_llm,
                tool_registry=registry,
            )
            
            result = await agent.research("Test question")
            
            assert result.answer == "Test answer"
            assert result.confidence == 0.9

    def test_fallback_models_config(self):
        """Test that fallback models are properly configured."""
        config = Config()
        config.llm.fallbacks = ["model-1", "model-2", "model-3"]
        
        client = create_llm_client(config.agent, config)
        assert client.fallback_models == ["model-1", "model-2", "model-3"]


class TestProviderConfigValidation:
    """Test that provider configurations are valid."""

    def test_config_has_required_llm_fields(self):
        """Test that config has all required LLM fields."""
        config = Config()
        assert hasattr(config.llm, 'model')
        assert hasattr(config.llm, 'small_model')
        assert hasattr(config.llm, 'fallbacks')
        assert config.llm.model is not None
        assert config.llm.small_model is not None

    def test_config_has_required_agent_fields(self):
        """Test that config has all required agent fields."""
        config = Config()
        assert hasattr(config.agent, 'tool_mode')
        assert hasattr(config.agent, 'max_steps')
        assert hasattr(config.agent, 'max_fetches')
        assert hasattr(config.agent, 'max_search_calls')
        assert hasattr(config.agent, 'max_tokens_total')
        assert hasattr(config.agent, 'wall_clock_seconds')
        assert hasattr(config.agent, 'stall_steps')
        assert hasattr(config.agent, 'trim_keep_last')

    def test_tool_mode_values(self):
        """Test that tool_mode accepts valid values."""
        config = Config()
        for mode in ["auto", "native", "json"]:
            config.agent.tool_mode = mode
            client = create_llm_client(config.agent, config)
            assert client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])