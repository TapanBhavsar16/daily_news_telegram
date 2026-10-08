from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from newsagent.tools.tinyfish import (
    TinyFishClient,
    SearchResult,
    SearchResponse,
    FetchResult,
    FetchResponse,
    Link,
    FetchError,
)
from newsagent.tools.registry import (
    ToolRegistry,
    Budget,
    DomainPolicy,
)
from newsagent.config import AgentConfig, CrawlConfig, Config


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
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def tinyfish_client(mock_httpx_client):
    client = TinyFishClient(api_key="test-key")
    client._client = mock_httpx_client
    return client


@pytest.mark.asyncio
async def test_search_success(tinyfish_client, mock_httpx_client):
    mock_httpx_client.get.return_value = MockHTTPResponse({
        "results": [
            {"title": "Test Result 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"title": "Test Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
        ],
        "total_results": 2,
    })

    result = await tinyfish_client.search("test query", num_results=5)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 2
    assert result.results[0].title == "Test Result 1"
    assert result.results[0].url == "https://example.com/1"
    assert result.query == "test query"
    assert result.total_results == 2


@pytest.mark.asyncio
async def test_search_caching(tinyfish_client, mock_httpx_client):
    mock_httpx_client.get.return_value = MockHTTPResponse({
        "results": [{"title": "Cached", "url": "https://cached.com", "snippet": "Cached"}],
        "total_results": 1,
    })

    result1 = await tinyfish_client.search("same query")
    result2 = await tinyfish_client.search("same query")

    assert result1.results[0].title == "Cached"
    assert result2.results[0].title == "Cached"
    assert mock_httpx_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_success(tinyfish_client, mock_httpx_client):
    mock_httpx_client.post.return_value = MockHTTPResponse({
        "results": [
            {
                "url": "https://example.com/page",
                "title": "Page Title",
                "text": "Page content here with [link](https://example.com/link)",
                "content_type": "text/html",
                "status_code": 200,
            }
        ],
        "errors": [],
    })

    result = await tinyfish_client.fetch(["https://example.com/page"])

    assert isinstance(result, FetchResponse)
    assert len(result.results) == 1
    assert result.results[0].title == "Page Title"
    assert result.results[0].url == "https://example.com/page"
    assert "Page content here" in result.results[0].text
    assert len(result.results[0].outbound_links) == 1
    assert result.results[0].outbound_links[0].url == "https://example.com/link"
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_fetch_with_error(tinyfish_client, mock_httpx_client):
    mock_httpx_client.post.return_value = MockHTTPResponse({
        "results": [
            {"url": "https://example.com/bad", "error": "404 Not Found"}
        ],
        "errors": [{"url": "https://example.com/bad", "error": "404 Not Found"}],
    })

    result = await tinyfish_client.fetch(["https://example.com/bad"])

    assert len(result.results) == 0
    assert len(result.errors) == 1
    assert result.errors[0].url == "https://example.com/bad"
    assert result.errors[0].error == "404 Not Found"


@pytest.mark.asyncio
async def test_link_extraction_markdown(tinyfish_client):
    text = "Check out [Google](https://google.com) and [GitHub](https://github.com)"
    links = tinyfish_client._extract_links(text, "https://example.com")

    assert len(links) == 2
    assert links[0].url == "https://google.com"
    assert links[0].anchor_text == "Google"
    assert links[1].url == "https://github.com"
    assert links[1].anchor_text == "GitHub"


@pytest.mark.asyncio
async def test_link_extraction_html(tinyfish_client):
    text = '<a href="https://example.com/link">Link</a>'
    links = tinyfish_client._extract_links(text, "https://example.com")

    assert len(links) == 1
    assert links[0].url == "https://example.com/link"


@pytest.mark.asyncio
async def test_blocked_domains_filtered(tinyfish_client):
    text = "Check [Twitter](https://twitter.com/user) and [Example](https://example.com)"
    links = tinyfish_client._extract_links(text, "https://example.com")

    assert len(links) == 1
    assert links[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_domain_limit_check():
    config = CrawlConfig(max_pages_per_domain=2)
    policy = DomainPolicy()
    registry = ToolRegistry(
        agent_config=AgentConfig(),
        crawl_config=config,
    )

    # Check limits BEFORE recording (simulating actual usage)
    assert registry._check_domain_limit("https://example.com/page1")  # count=0 < 2
    assert registry._check_domain_limit("https://example.com/page2")  # count=0 < 2
    assert registry._check_domain_limit("https://example.com/page3")  # count=0 < 2

    # Now record fetches
    registry._record_domain_fetch("https://example.com/page1")  # count=1
    registry._record_domain_fetch("https://example.com/page2")  # count=2

    # After 2 fetches, limit reached
    assert not registry._check_domain_limit("https://example.com/page3")  # count=2 !< 2


def test_budget_exhausted():
    budget = Budget(max_steps=2, max_fetches=1, max_search_calls=1)
    assert not budget.exhausted()

    budget.record_step()
    budget.record_step()
    assert budget.exhausted()


def test_budget_stalled():
    budget = Budget(stall_steps=2)
    budget.record_step()
    budget.record_step()
    budget.record_step()
    assert budget.stalled()

    budget.record_fact()
    budget.record_step()
    budget.record_step()
    assert not budget.stalled()


def test_tool_registry_search_budget(mock_httpx_client):
    config = AgentConfig(max_search_calls=1)
    crawl_config = CrawlConfig()
    registry = ToolRegistry(config, crawl_config)

    # Mock the tinyfish client
    mock_tinyfish = AsyncMock()
    mock_tinyfish.search.return_value = SearchResponse(
        results=[SearchResult(title="Test", url="https://test.com", snippet="Test")],
        query="test",
    )
    registry.tinyfish_client = mock_tinyfish

    import asyncio
    async def test():
        result1 = await registry._execute_search({"query": "test"})
        result2 = await registry._execute_search({"query": "test again"})
        return result1, result2

    result1, result2 = asyncio.run(test())
    data1 = json.loads(result1)
    data2 = json.loads(result2)
    assert "results" in data1
    assert "error" in data2
    assert "budget exhausted" in data2["error"]


def test_tool_registry_duplicate_query(mock_httpx_client):
    config = AgentConfig()
    crawl_config = CrawlConfig()
    registry = ToolRegistry(config, crawl_config)

    # Mock the tinyfish client
    mock_tinyfish = AsyncMock()
    mock_tinyfish.search.return_value = SearchResponse(
        results=[SearchResult(title="Test", url="https://test.com", snippet="Test")],
        query="duplicate test",
    )
    registry.tinyfish_client = mock_tinyfish

    import asyncio
    async def test():
        result1 = await registry._execute_search({"query": "duplicate test"})
        result2 = await registry._execute_search({"query": "duplicate test"})
        return result1, result2

    result1, result2 = asyncio.run(test())
    data1 = json.loads(result1)
    data2 = json.loads(result2)
    assert "results" in data1
    assert data2.get("cached") is True


def test_tool_registry_fetch_dedupe():
    config = AgentConfig()
    crawl_config = CrawlConfig()
    registry = ToolRegistry(config, crawl_config)

    import asyncio
    async def test():
        result1 = await registry._execute_fetch({"url": "https://example.com/page"})
        result2 = await registry._execute_fetch({"url": "https://example.com/page"})
        return result1, result2

    result1, result2 = asyncio.run(test())
    data1 = json.loads(result1)
    data2 = json.loads(result2)
    assert "error" in data1
    assert data2.get("cached") is True


def test_tool_registry_domain_blocked():
    config = AgentConfig()
    crawl_config = CrawlConfig(blocked_domains=["blocked.com"])
    registry = ToolRegistry(config, crawl_config)

    import asyncio
    async def test():
        return await registry._execute_fetch({"url": "https://blocked.com/page"})

    result = asyncio.run(test())
    data = json.loads(result)
    assert "error" in data
    assert "not allowed" in data["error"]


def test_budget_summary():
    budget = Budget(
        max_steps=10, max_fetches=5, max_search_calls=3,
        max_tokens_total=1000, wall_clock_seconds=60
    )
    budget.record_step()
    budget.record_fetch()
    budget.record_search()
    budget.record_tokens(100)

    summary = budget.remaining_summary()
    assert "Steps: 1/10" in summary
    assert "Fetches: 1/5" in summary
    assert "Searches: 1/3" in summary
    assert "Tokens: 100/1000" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])