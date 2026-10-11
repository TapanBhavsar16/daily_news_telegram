from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

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
    outbound_links: list[Link] = Field(default_factory=list)
    content_type: str = "text/html"
    status_code: int = 200


class Link(BaseModel):
    url: str
    anchor_text: str = ""


class FetchResponse(BaseModel):
    results: list[FetchResult] = Field(default_factory=list)
    errors: list[FetchError] = Field(default_factory=list)


class FetchError(BaseModel):
    url: str
    error: str


@dataclass
class TinyFishClient:
    api_key: str
    base_url: str = "https://api.tinyfish.io/v1"
    timeout: float = 30.0
    _cache: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
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

        params = {
            "query": query,
            "num_results": num_results,
            "location": location,
        }
        if recency_days:
            params["recency_days"] = recency_days

        try:
            response = await self._client.get("/search", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limited by TinyFish search", query=query)
                raise
            raise

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                position=i,
            )
            for i, r in enumerate(data.get("results", []))
        ]

        result = SearchResponse(
            results=results,
            query=query,
            total_results=data.get("total_results", len(results)),
        )
        self._cache[cache_key] = result
        logger.info("Search completed", query=query, results=len(results))
        return result

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
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

        try:
            response = await self._client.post(
                "/fetch",
                json={"urls": urls, "format": format, "max_chars": max_chars},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limited by TinyFish fetch", urls=urls)
                raise
            raise

        results = []
        errors = []

        for item in data.get("results", []):
            url = item.get("url", "")
            if "error" in item:
                errors.append(FetchError(url=url, error=item["error"]))
                continue

            text = item.get("text", "") or item.get("markdown", "") or ""
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"

            outbound_links = self._extract_links(text, url)

            results.append(
                FetchResult(
                    title=item.get("title", ""),
                    url=url,
                    text=text,
                    outbound_links=outbound_links,
                    content_type=item.get("content_type", "text/html"),
                    status_code=item.get("status_code", 200),
                )
            )

        result = FetchResponse(results=results, errors=errors)
        self._cache[cache_key] = result
        logger.info("Fetch completed", urls=len(urls), success=len(results), errors=len(errors))
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
        await self._client.aclose()

    def clear_cache(self):
        self._cache.clear()


def create_tinyfish_client(config: Config = None) -> TinyFishClient:
    settings = get_settings()
    api_key = get_tinyfish_api_key(settings)
    return TinyFishClient(api_key=api_key)


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