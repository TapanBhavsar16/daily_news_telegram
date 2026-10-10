from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from newsagent.config import Config, AgentConfig, CrawlConfig
from newsagent.tools import (
    ToolRegistry,
    TinyFishClient,
    TelegramClient,
    SearchResponse,
    SearchResult,
    FetchResponse,
    FetchResult,
    Link,
    create_tool_registry,
    create_tinyfish_client,
    create_telegram_client,
    format_story_message,
    escape_html,
)
from newsagent.tools.tinyfish import search_web, fetch_pages


class MockHTTPResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_httpx():
    with patch("httpx.AsyncClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def tool_registry(config, mock_httpx):
    mock_tinyfish = AsyncMock()
    mock_tinyfish.search.return_value = SearchResponse(
        results=[
            SearchResult(title="AI News 1", url="https://ai-news.com/1", snippet="AI breakthrough"),
            SearchResult(title="AI News 2", url="https://ai-news.com/2", snippet="New model released"),
        ],
        query="latest AI news",
    )
    mock_tinyfish.fetch.return_value = FetchResponse(
        results=[
            FetchResult(
                title="AI News 1",
                url="https://ai-news.com/1",
                text="Full article content about AI breakthrough with [source](https://official-ai.com/announcement)",
                outbound_links=[Link(url="https://official-ai.com/announcement", anchor_text="source")],
            )
        ],
        errors=[],
    )
    
    registry = create_tool_registry(config)
    registry.tinyfish_client = mock_tinyfish
    return registry


@pytest.mark.asyncio
async def test_full_research_flow(tool_registry):
    """Test a complete research flow: search -> fetch -> scratchpad -> finish"""
    
    # 1. Search
    search_result = await tool_registry.guarded_execute("search", {"query": "latest AI news"})
    search_data = json.loads(search_result)
    assert "results" in search_data
    assert len(search_data["results"]) == 2
    
    # 2. Fetch first result
    fetch_result = await tool_registry.guarded_execute("fetch", {"url": "https://ai-news.com/1"})
    fetch_data = json.loads(fetch_result)
    assert "text" in fetch_data
    assert "outbound_links" in fetch_data
    assert len(fetch_data["outbound_links"]) == 1
    
    # 3. Use scratchpad to record findings
    class MockScratchpad:
        def __init__(self):
            self.notes = {}
        def run(self, action, key="", content="", source_url=""):
            if action == "write":
                self.notes[key] = {"content": content, "source_url": source_url}
            elif action == "append":
                if key in self.notes:
                    self.notes[key]["content"] += content
                else:
                    self.notes[key] = {"content": content, "source_url": source_url}
            elif action == "read":
                return self.notes.get(key, {}).get("content", "")
            elif action == "list":
                return list(self.notes.keys())
            elif action == "delete":
                self.notes.pop(key, None)
    
    scratchpad = MockScratchpad()
    
    # Write plan
    sp_result = await tool_registry.guarded_execute("scratchpad", {
        "action": "write",
        "key": "plan",
        "content": "Research AI news"
    }, scratchpad=scratchpad)
    sp_data = json.loads(sp_result)
    assert sp_data["status"] == "written"
    
    # Write fact with source
    sp_result = await tool_registry.guarded_execute("scratchpad", {
        "action": "write",
        "key": "facts_1",
        "content": "AI breakthrough announced",
        "source_url": "https://ai-news.com/1"
    }, scratchpad=scratchpad)
    sp_data = json.loads(sp_result)
    assert sp_data["status"] == "written"
    
    # 4. Finish
    finish_result = await tool_registry.guarded_execute("finish", {
        "answer": "AI breakthrough announced with new model",
        "confidence": 0.9,
        "open_gaps": []
    })
    finish_data = json.loads(finish_result)
    assert finish_data["status"] == "finish"
    assert finish_data["args"]["answer"] == "AI breakthrough announced with new model"


@pytest.mark.asyncio
async def test_budget_enforcement(config):
    """Test that budgets are properly enforced across multiple operations"""
    mock_tinyfish = AsyncMock()
    mock_tinyfish.search.return_value = SearchResponse(
        results=[SearchResult(title="Test", url="https://test.com", snippet="Test")],
        query="test",
    )
    mock_tinyfish.fetch.return_value = FetchResponse(
        results=[FetchResult(title="Test", url="https://test.com", text="Content")],
        errors=[],
    )
    
    # Create registry with tight budgets
    tight_config = Config()
    tight_config.agent.max_search_calls = 2
    tight_config.agent.max_fetches = 2
    tight_config.agent.max_steps = 5
    
    registry = create_tool_registry(tight_config)
    registry.tinyfish_client = mock_tinyfish
    
    # Exhaust search budget
    for i in range(3):
        result = await registry.guarded_execute("search", {"query": f"query {i}"})
        data = json.loads(result)
        if i < 2:
            assert "results" in data
        else:
            assert "error" in data
            assert "budget exhausted" in data["error"]


@pytest.mark.asyncio
async def test_deduplication(tool_registry):
    """Test that duplicate URLs and queries are deduplicated"""
    
    # First search
    result1 = await tool_registry.guarded_execute("search", {"query": "duplicate test"})
    data1 = json.loads(result1)
    assert "results" in data1
    
    # Duplicate search
    result2 = await tool_registry.guarded_execute("search", {"query": "duplicate test"})
    data2 = json.loads(result2)
    assert data2.get("cached") is True
    
    # First fetch - use a different URL to avoid the mock
    result1 = await tool_registry.guarded_execute("fetch", {"url": "https://different-site.com/page"})
    data1 = json.loads(result1)
    # This will fail because the mock is set up for ai-news.com, but that's fine
    # we just want to test dedupe logic works
    
    # Duplicate fetch
    result2 = await tool_registry.guarded_execute("fetch", {"url": "https://different-site.com/page"})
    data2 = json.loads(result2)
    assert data2.get("cached") is True


@pytest.mark.asyncio
async def test_domain_policy_enforcement(config):
    """Test that blocked domains are rejected"""
    config.crawl.blocked_domains = ["blocked-site.com"]
    mock_tinyfish = AsyncMock()
    
    registry = create_tool_registry(config)
    registry.tinyfish_client = mock_tinyfish
    
    result = await registry.guarded_execute("fetch", {"url": "https://blocked-site.com/page"})
    data = json.loads(result)
    assert "error" in data
    assert "not allowed" in data["error"]


@pytest.mark.asyncio
async def test_telegram_formatting():
    """Test Telegram message formatting"""
    
    stories = [
        {
            "headline": "AI Breakthrough",
            "summary": "New model achieves SOTA",
            "key_details": ["10x faster", "Open source"],
            "why_it_matters": "Democratizes AI",
            "sources": [
                {"url": "https://source1.com", "label": "Source 1"},
                {"url": "https://source2.com", "label": "Source 2"},
            ],
            "date": "8 Oct 2026",
        }
    ]
    
    message = format_story_message("AI", "8 Oct 2026", stories)
    
    assert "AI Breakthrough" in message
    assert "New model achieves SOTA" in message
    assert "10x faster" in message
    assert "Open source" in message
    assert "Why it matters" in message
    assert "Democratizes AI" in message
    assert "Source 1" in message
    assert "Source 2" in message
    assert "8 Oct 2026" in message


@pytest.mark.asyncio
async def test_html_escaping():
    """Test HTML escaping for Telegram"""
    assert escape_html("<script>alert('xss')</script>") == "&lt;script&gt;alert('xss')&lt;/script&gt;"
    assert escape_html("A & B") == "A &amp; B"
    assert escape_html('Quote "test"') == 'Quote "test"'


@pytest.mark.asyncio
async def test_message_splitting():
    """Test long message splitting"""
    client = TelegramClient(
        bot_token="test:token",
        chat_id="123456",
    )
    
    # Create a long message
    long_text = "Line\n" * 1000  # Well over 4096 chars
    parts = client._split_message(long_text)
    
    assert len(parts) > 1
    for part in parts:
        assert len(part) <= 4096


@pytest.mark.asyncio
async def test_tool_schemas(config):
    """Test that tool schemas are correctly generated"""
    mock_tinyfish = AsyncMock()
    registry = create_tool_registry(config)
    registry.tinyfish_client = mock_tinyfish
    
    schemas = registry.get_tool_schemas()
    
    assert len(schemas) == 4
    names = [s["function"]["name"] for s in schemas]
    assert "search" in names
    assert "fetch" in names
    assert "scratchpad" in names
    assert "finish" in names
    
    # Check search schema
    search_schema = next(s for s in schemas if s["function"]["name"] == "search")
    assert "query" in search_schema["function"]["parameters"]["properties"]
    assert "num_results" in search_schema["function"]["parameters"]["properties"]
    
    # Check fetch schema
    fetch_schema = next(s for s in schemas if s["function"]["name"] == "fetch")
    assert "url" in fetch_schema["function"]["parameters"]["properties"]
    
    # Check finish schema
    finish_schema = next(s for s in schemas if s["function"]["name"] == "finish")
    assert "answer" in finish_schema["function"]["parameters"]["properties"]
    assert "confidence" in finish_schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_idempotent_send_crash_recovery():
    """Test that pending posts can be recovered after a crash."""
    from newsagent.config import Config
    from newsagent.memory.store import create_memory_store
    
    config = Config()
    memory = create_memory_store(config)
    
    # Use unique topic to avoid conflicts with other tests
    topic = "Test Topic Crash Recovery"
    
    # Simulate creating a pending post (before send)
    pending_id = memory.create_pending_post(
        topic=topic,
        message="<b>Test Message</b>",
        items=[{"candidate": {"title": "Test"}, "answer": "Test answer", "sources": ["https://example.com"]}],
    )
    
    # Verify pending post exists and is unsent
    unsent = memory.get_unsent_pending(topic)
    assert len(unsent) == 1
    assert unsent[0].id == pending_id
    assert unsent[0].sent_at is None
    
    # Simulate successful send - mark as sent
    result = memory.mark_pending_sent(pending_id, telegram_message_id=12345)
    assert result is True
    
    # Verify no unsent posts remain
    unsent = memory.get_unsent_pending(topic)
    assert len(unsent) == 0


@pytest.mark.asyncio
async def test_duplicate_pending_post_not_created_on_resume():
    """Test that resuming doesn't create duplicate pending posts."""
    from newsagent.config import Config
    from newsagent.memory.store import create_memory_store
    
    config = Config()
    memory = create_memory_store(config)
    
    # Use unique topic to avoid conflicts with other tests
    topic = "Test Topic Resume"
    
    # Create a pending post and mark it as sent (simulating completed send)
    pending_id = memory.create_pending_post(
        topic=topic,
        message="<b>Test Message</b>",
        items=[{"candidate": {"title": "Test"}, "answer": "Test answer", "sources": ["https://example.com"]}],
    )
    memory.mark_pending_sent(pending_id, telegram_message_id=12345)
    
    # On resume, check for unsent posts - should be empty
    unsent = memory.get_unsent_pending(topic)
    assert len(unsent) == 0
    
    # The posted items should not exist yet (add_posted_item wasn't called)
    recent = memory.get_recent_posts(topic, days=1)
    assert len(recent) == 0


@pytest.mark.asyncio
async def test_full_posted_item_persistence():
    """Test that posted items are properly saved after successful send."""
    from newsagent.config import Config
    from newsagent.memory.store import create_memory_store
    import uuid
    
    config = Config()
    memory = create_memory_store(config)
    
    # Use unique topic and URL to avoid conflicts with other tests
    unique_id = str(uuid.uuid4())[:8]
    topic = f"Test Topic Full Persistence {unique_id}"
    url = f"https://example.com/persistence/{unique_id}"
    
    # Simulate the full flow: pending -> sent -> posted items saved
    pending_id = memory.create_pending_post(
        topic=topic,
        message="<b>Test Message</b>",
        items=[{"candidate": {"title": "Test Story", "url": url}, "answer": "Test answer", "sources": [url]}],
    )
    
    # Mark as sent
    memory.mark_pending_sent(pending_id, telegram_message_id=12345)
    
    # Add posted items (as done in post_telegram_node after successful send)
    item_id = memory.add_posted_item(
        topic=topic,
        title="Test Story",
        url=url,
        summary="Test answer",
        key_facts=["Fact 1", "Fact 2"],
        entities=["Entity 1"],
        run_id="test_run_123",
    )
    
    # Verify posted item exists
    recent = memory.get_recent_posts(topic, days=1)
    assert len(recent) == 1
    assert recent[0].title == "Test Story"
    assert recent[0].url == url
    assert "Fact 1" in recent[0].key_facts
    
    # Verify duplicate detection works
    is_dup, reason = memory.check_duplicate(
        url=url,
        title="Test Story",
        summary="Test answer",
        key_facts=["Fact 1", "Fact 2"],
    )
    assert is_dup is True
    assert "URL already posted" in reason or "Title already posted" in reason or "Content already posted" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
