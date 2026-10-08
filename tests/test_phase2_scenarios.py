"""Executable Phase 2 scenarios.

These tests model a real research run while keeping TinyFish, Telegram, and
the LLM out of the network.  The credentials in ``.env`` are intentionally
not read by these tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from newsagent.config import Config, load_config
from newsagent.tools.registry import create_tool_registry
from newsagent.tools.telegram import TelegramClient, format_story_message
from newsagent.tools.tinyfish import (
    FetchResponse,
    FetchResult,
    Link,
    SearchResponse,
    SearchResult,
)


@dataclass
class ScenarioScratchpad:
    """Small in-memory implementation of the Phase 2 scratchpad contract."""

    notes: dict[str, dict[str, str]] = field(default_factory=dict)

    def run(self, action: str, key: str = "", content: str = "", source_url: str = ""):
        if action == "write":
            self.notes[key] = {"content": content, "source_url": source_url}
        elif action == "append":
            note = self.notes.setdefault(key, {"content": "", "source_url": source_url})
            note["content"] += content
            if source_url:
                note["source_url"] = source_url
        elif action == "read":
            return self.notes.get(key, {}).get("content", "")
        elif action == "list":
            return list(self.notes)
        elif action == "delete":
            self.notes.pop(key, None)
        else:
            raise ValueError(f"unsupported action: {action}")


def make_registry(*, max_search_calls: int = 6, max_fetches: int = 8):
    config = Config()
    config.agent.max_search_calls = max_search_calls
    config.agent.max_fetches = max_fetches
    registry = create_tool_registry(config)
    registry.tinyfish_client = AsyncMock()
    return registry


@pytest.mark.asyncio
async def test_realistic_news_research_run_records_cited_facts_and_finishes():
    """A daily-news candidate is searched, verified, noted, and finalized."""

    registry = make_registry()
    registry.tinyfish_client.search.return_value = SearchResponse(
        query="quantum computing funding October 2026",
        results=[
            SearchResult(
                title="Quantum company announces new funding",
                url="https://news.example/story",
                snippet="The company announced a $50m round.",
            ),
            SearchResult(
                title="Industry reaction to quantum funding",
                url="https://analysis.example/reaction",
                snippet="Analysts discuss the wider market impact.",
            ),
        ],
    )
    registry.tinyfish_client.fetch.return_value = FetchResponse(
        results=[
            FetchResult(
                title="Official funding announcement",
                url="https://company.example/press/funding",
                text="The company announced a $50m Series B on 8 October 2026.",
                outbound_links=[
                    Link(
                        url="https://company.example/press/funding-details",
                        anchor_text="funding details",
                    )
                ],
            )
        ]
    )
    pad = ScenarioScratchpad()

    search = json.loads(
        await registry.guarded_execute(
            "search",
            {"query": "quantum computing funding October 2026", "num_results": 5},
        )
    )
    assert search["results"][0]["url"] == "https://news.example/story"

    fetched = json.loads(
        await registry.guarded_execute(
            "fetch", {"url": "https://company.example/press/funding"}
        )
    )
    assert "$50m" in fetched["text"]
    assert fetched["outbound_links"][0]["url"].endswith("funding-details")

    await registry.guarded_execute(
        "scratchpad",
        {
            "action": "write",
            "key": "facts_1",
            "content": "$50m Series B announced on 8 October 2026",
            "source_url": fetched["url"],
        },
        scratchpad=pad,
    )
    await registry.guarded_execute(
        "scratchpad",
        {"action": "write", "key": "open_questions", "content": "Find independent reaction"},
        scratchpad=pad,
    )
    finished = json.loads(
        await registry.guarded_execute(
            "finish",
            {
                "answer": "The company announced a $50m Series B on 8 October 2026.",
                "confidence": 0.92,
                "open_gaps": [],
            },
        )
    )

    assert finished["status"] == "finish"
    assert pad.run("read", key="facts_1") == "$50m Series B announced on 8 October 2026"
    assert pad.run("read", key="open_questions") == "Find independent reaction"
    assert registry.budget.search_calls_used == 1
    assert registry.budget.fetches_used == 1


@pytest.mark.asyncio
async def test_research_agent_does_not_repeat_search_or_fetch_same_story():
    """Loop guards prevent duplicate work when a model repeats an action."""

    registry = make_registry()
    registry.tinyfish_client.search.return_value = SearchResponse(
        query="central bank rate decision",
        results=[SearchResult(title="Rate decision", url="https://example.com/rates", snippet="")],
    )
    registry.tinyfish_client.fetch.return_value = FetchResponse(
        results=[FetchResult(title="Rates", url="https://example.com/rates", text="Decision")]
    )

    first_search = json.loads(
        await registry.guarded_execute("search", {"query": "central bank rate decision"})
    )
    repeated_search = json.loads(
        await registry.guarded_execute("search", {"query": "CENTRAL BANK RATE DECISION"})
    )
    first_fetch = json.loads(
        await registry.guarded_execute(
            "fetch", {"url": "https://example.com/rates?utm_source=feed"}
        )
    )
    repeated_fetch = json.loads(await registry.guarded_execute("fetch", {"url": "https://example.com/rates"}))

    assert "results" in first_search
    assert repeated_search == {"error": "Duplicate query", "cached": True}
    assert first_fetch["text"] == "Decision"
    assert repeated_fetch["cached"] is True
    registry.tinyfish_client.search.assert_awaited_once()
    registry.tinyfish_client.fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_crawl_policy_stops_untrusted_domain_before_provider_call():
    """A blocked social/aggregator link is rejected before crawling it."""

    config = Config()
    config.crawl.blocked_domains = ["social.example"]
    registry = create_tool_registry(config)
    registry.tinyfish_client = AsyncMock()

    result = json.loads(
        await registry.guarded_execute("fetch", {"url": "https://social.example/post/123"})
    )

    assert result["error"] == "Domain not allowed"
    registry.tinyfish_client.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_digest_is_safe_for_html_and_telegram_length_limits():
    """Untrusted article text is escaped and a long digest is split safely."""

    story = {
        "headline": "<Breaking> & update",
        "summary": "A source says <unverified> details changed.",
        "key_details": ["Amount: $50m & rising"],
        "why_it_matters": "Readers need <context>.",
        "date": "8 Oct 2026",
        "sources": [{"url": "https://source.example/a?x=1&y=2", "label": "Official & source"}],
    }
    message = format_story_message("Technology & Science", "8 Oct 2026", [story])
    assert "&lt;Breaking&gt; &amp; update" in message
    assert 'x=1&amp;y=2' in message
    assert "<unverified>" not in message

    client = TelegramClient(bot_token="test:token", chat_id="123")
    parts = client._split_message("<b>Story</b>\n" * 900)
    assert len(parts) > 1
    assert all(len(part) <= 4096 for part in parts)


def test_phase2_config_loads_without_exposing_or_requiring_credentials():
    """The checked-in YAML drives behavior; secrets remain external config."""

    config = load_config()
    assert config.agent.tool_mode == "auto"
    assert config.news.max_discovery_rounds == 3
    assert config.crawl.max_pages_per_domain == 3
    assert config.tinyfish.api_key_env == "TINYFISH_API_KEY"
    assert config.telegram.bot_token_env == "TELEGRAM_BOT_TOKEN"
