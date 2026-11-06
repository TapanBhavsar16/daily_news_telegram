from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from tinyfish import TinyFish

from newsagent.config import Config, TinyFishConfig, get_settings, get_tinyfish_api_key
from newsagent.logging import get_logger

logger = get_logger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    position: int = 0


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    query: str = ""
    total_results: int = 0


class FetchResult(BaseModel):
    title: str
    url: str
    text: str
    outbound_links: list["Link"] = Field(default_factory=list)
    content_type: str = "text/html"
    status_code: int = 200


class Link(BaseModel):
    url: str
    anchor_text: str = ""


class FetchResponse(BaseModel):
    results: list[FetchResult] = Field(default_factory=list)
    errors: list["FetchError"] = Field(default_factory=list)


class FetchError(BaseModel):
    url: str
    error: str


@dataclass
class TinyFishClient:
    api_key: str
    base_url: str = "https://agent.tinyfish.ai"
    timeout: float = 30.0
    _cache: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._sdk = TinyFish(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def search(
        self,
        query: str,
        num_results: int = 10,
        location: str = "US",
        recency_days: Optional[int] = None,
    ) -> SearchResponse:
        cache_key = f"search:{query}:{num_results}:{location}:{recency_days}"
        if cache_key in self._cache:
            logger.debug("Cache hit for search", query=query)
            return self._cache[cache_key]

        recency_minutes = recency_days * 24 * 60 if recency_days else None

        def _call() -> SearchResponse:
            resp = self._sdk.search.query(
                query=query,
                location=location,
                recency_minutes=recency_minutes,
            )
            results = [
                SearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    position=i,
                )
                for i, r in enumerate(resp.results)
            ]
            return SearchResponse(
                results=results,
                query=resp.query,
                total_results=resp.total_results,
            )

        result = await asyncio.to_thread(_call)
        self._cache[cache_key] = result
        logger.info("Search completed", query=query, results=len(result.results))
        return result

    async def fetch(
        self,
        urls: list[str],
        format: str = "markdown",
        max_chars: int = 8000,
    ) -> FetchResponse:
        cache_key = f"fetch:{','.join(sorted(urls))}:{format}:{max_chars}"
        if cache_key in self._cache:
            logger.debug("Cache hit for fetch", urls=urls)
            return self._cache[cache_key]

        def _call() -> FetchResponse:
            resp = self._sdk.fetch.get_contents(
                urls=urls,
                format=format,
            )
            results = []
            errors = []

            for item in resp.results:
                url = item.url
                text = item.text or ""
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"

                outbound_links = self._extract_links(text, url)

                results.append(
                    FetchResult(
                        title=item.title or "",
                        url=url,
                        text=text,
                        outbound_links=outbound_links,
                        content_type=item.format or "text/html",
                        status_code=200,
                    )
                )

            for item in resp.errors:
                errors.append(FetchError(url=item.url, error=item.error))

            return FetchResponse(results=results, errors=errors)

        result = await asyncio.to_thread(_call)
        self._cache[cache_key] = result
        logger.info("Fetch completed", urls=len(urls), success=len(result.results), errors=len(result.errors))
        return result

    def _extract_links(self, text: str, base_url: str) -> list[Link]:
        links = []
        try:
            base_domain = urlparse(base_url).netloc

            markdown_links = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)
            for anchor, url in markdown_links:
                if self._is_valid_link(url, base_domain):
                    links.append(Link(url=url, anchor_text=anchor))

            html_links = re.findall(r'href=["\'](https?://[^"\']+)["\']', text)
            for url in html_links:
                if self._is_valid_link(url, base_domain):
                    if not any(l.url == url for l in links):
                        links.append(Link(url=url, anchor_text=""))
        except Exception as e:
            logger.warning("Link extraction failed", error=str(e))

        return links[:50]

    def _is_valid_link(self, url: str, base_domain: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if parsed.netloc == base_domain:
                return True
            blocked = {
                "twitter.com",
                "x.com",
                "facebook.com",
                "instagram.com",
                "linkedin.com",
                "youtube.com",
                "tiktok.com",
                "reddit.com",
                "pinterest.com",
            }
            if any(b in parsed.netloc for b in blocked):
                return False
            return True
        except Exception:
            return False

    async def close(self):
        return None

    def clear_cache(self):
        self._cache.clear()


def create_tinyfish_client(config: Config = None) -> TinyFishClient:
    settings = get_settings()
    api_key = get_tinyfish_api_key(settings)
    base_url = config.tinyfish.base_url if config else None
    return TinyFishClient(
        api_key=api_key,
        base_url=base_url,
    )


async def search_web(
    query: str,
    num_results: int = 10,
    location: str = "US",
    recency_days: Optional[int] = None,
    client: Optional[TinyFishClient] = None,
) -> SearchResponse:
    if client is None:
        client = create_tinyfish_client()
    return await client.search(query, num_results, location, recency_days)


async def fetch_pages(
    urls: list[str],
    format: str = "markdown",
    max_chars: int = 8000,
    client: Optional[TinyFishClient] = None,
) -> FetchResponse:
    if client is None:
        client = create_tinyfish_client()
    return await client.fetch(urls, format, max_chars)
