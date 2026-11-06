from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from newsagent.config import AgentConfig, CrawlConfig, Config
from newsagent.logging import get_logger
from newsagent.tools.tinyfish import TinyFishClient, search_web, fetch_pages

logger = get_logger(__name__)


@dataclass
class Budget:
    max_steps: int = 20
    max_fetches: int = 8
    max_search_calls: int = 6
    max_tokens_total: int = 120000
    wall_clock_seconds: int = 180
    stall_steps: int = 3

    steps_used: int = 0
    fetches_used: int = 0
    search_calls_used: int = 0
    tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    start_time: float = field(default_factory=time.time)
    last_fact_step: int = 0
    llm_calls: int = 0

    def exhausted(self) -> bool:
        return (
            self.steps_used >= self.max_steps
            or self.fetches_used >= self.max_fetches
            or self.search_calls_used >= self.max_search_calls
            or self.tokens_used >= self.max_tokens_total
            or (time.time() - self.start_time) >= self.wall_clock_seconds
        )

    def stalled(self, stall_steps: int = 3) -> bool:
        return (self.steps_used - self.last_fact_step) >= stall_steps

    def record_step(self):
        self.steps_used += 1

    def record_fetch(self):
        self.fetches_used += 1

    def record_search(self):
        self.search_calls_used += 1

    def record_tokens(self, tokens: int):
        self.tokens_used += tokens

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens
        self.tokens_used += total_tokens
        self.llm_calls += 1

    def record_fact(self):
        self.last_fact_step = self.steps_used

    def remaining_summary(self) -> str:
        return (
            f"Steps: {self.steps_used}/{self.max_steps}, "
            f"Fetches: {self.fetches_used}/{self.max_fetches}, "
            f"Searches: {self.search_calls_used}/{self.max_search_calls}, "
            f"Tokens: {self.tokens_used}/{self.max_tokens_total} (prompt: {self.prompt_tokens_used}, completion: {self.completion_tokens_used}), "
            f"LLM calls: {self.llm_calls}, "
            f"Time: {time.time() - self.start_time:.0f}/{self.wall_clock_seconds}s"
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.start_time)

    def cost_summary(self) -> dict[str, Any]:
        """Return cost summary for logging."""
        return {
            "total_tokens": self.tokens_used,
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "llm_calls": self.llm_calls,
            "steps_used": self.steps_used,
            "fetches_used": self.fetches_used,
            "search_calls_used": self.search_calls_used,
            "wall_time_seconds": time.time() - self.start_time,
        }


@dataclass
class DomainPolicy:
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    def is_allowed(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            if self.blocked_domains and any(b in domain for b in self.blocked_domains):
                return False
            if self.allowed_domains and not any(a in domain for a in self.allowed_domains):
                return False
            return True
        except Exception:
            return True


class ToolRegistry:
    def __init__(
        self,
        agent_config: AgentConfig,
        crawl_config: CrawlConfig,
        tinyfish_client: Optional[TinyFishClient] = None,
    ):
        self.agent_config = agent_config
        self.crawl_config = crawl_config
        self.tinyfish_client = tinyfish_client
        self.budget = Budget(
            max_steps=agent_config.max_steps,
            max_fetches=agent_config.max_fetches,
            max_search_calls=agent_config.max_search_calls,
            max_tokens_total=agent_config.max_tokens_total,
            wall_clock_seconds=agent_config.wall_clock_seconds,
            stall_steps=agent_config.stall_steps,
        )
        self.domain_policy = DomainPolicy(
            allowed_domains=crawl_config.allowed_domains,
            blocked_domains=crawl_config.blocked_domains,
        )
        self.visited_urls: set[str] = set()
        self.seen_queries: set[str] = set()
        self.domain_fetch_count: dict[str, int] = {}

    def _canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        except Exception:
            return url

    def _check_domain_limit(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc
            count = self.domain_fetch_count.get(domain, 0)
            return count < self.crawl_config.max_pages_per_domain
        except Exception:
            return True

    def _record_domain_fetch(self, url: str):
        try:
            domain = urlparse(url).netloc
            self.domain_fetch_count[domain] = self.domain_fetch_count.get(domain, 0) + 1
        except Exception:
            pass

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information. Use specific, focused queries (2-6 words).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default 10)",
                                "default": 10,
                            },
                            "recency_days": {
                                "type": "integer",
                                "description": "Only return results from the last N days",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch",
                    "description": "Fetch full content of a URL. Returns cleaned text, title, and extracted outbound links.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scratchpad",
                    "description": "Read/write persistent notes. Use for plans, checklists, facts, conflicts, open questions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["write", "append", "read", "list", "delete"],
                            },
                            "key": {"type": "string", "description": "Note key (e.g., 'plan', 'checklist', 'facts_1')"},
                            "content": {"type": "string", "description": "Content to write/append"},
                            "source_url": {"type": "string", "description": "Source URL for fact notes"},
                        },
                        "required": ["action", "key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finish research and return final answer with citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Final answer in own words"},
                            "confidence": {
                                "type": "number",
                                "description": "Confidence 0.0-1.0",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "open_gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Unanswered questions",
                            },
                        },
                        "required": ["answer", "confidence"],
                    },
                },
            },
        ]

    async def guarded_execute(
        self,
        name: str,
        args: dict[str, Any],
        scratchpad: Optional[Any] = None,
    ) -> str:
        self.budget.record_step()

        if name == "search":
            return await self._execute_search(args)
        elif name == "fetch":
            return await self._execute_fetch(args)
        elif name == "scratchpad":
            return self._execute_scratchpad(args, scratchpad)
        elif name == "finish":
            return json.dumps({"status": "finish", "args": args})
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    async def _execute_search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        num_results = args.get("num_results", 10)

        if not query:
            return json.dumps({"error": "Empty query"})

        query_lower = query.lower()
        if query_lower in self.seen_queries:
            return json.dumps({"error": "Duplicate query", "cached": True})

        if self.budget.search_calls_used >= self.budget.max_search_calls:
            return json.dumps(
                {"error": f"Search budget exhausted ({self.budget.max_search_calls} calls)"}
            )

        self.seen_queries.add(query_lower)
        self.budget.record_search()

        try:
            response = await search_web(
                query=query,
                num_results=min(num_results, 10),
                recency_days=args.get("recency_days"),
                client=self.tinyfish_client,
            )

            results = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "position": r.position,
                }
                for r in response.results
            ]
            return json.dumps({"results": results, "query": query})
        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            return json.dumps({"error": f"Search failed: {str(e)}"})

    async def _execute_fetch(self, args: dict[str, Any]) -> str:
        url = args.get("url", "").strip()
        if not url:
            return json.dumps({"error": "Empty URL"})

        canonical = self._canonical_url(url)
        if canonical in self.visited_urls:
            return json.dumps({"error": "Already fetched", "cached": True, "url": url})

        if not self.domain_policy.is_allowed(url):
            return json.dumps({"error": "Domain not allowed", "url": url})

        if not self._check_domain_limit(url):
            return json.dumps(
                {"error": f"Domain fetch limit reached ({self.crawl_config.max_pages_per_domain})", "url": url}
            )

        if self.budget.fetches_used >= self.budget.max_fetches:
            return json.dumps(
                {"error": f"Fetch budget exhausted ({self.budget.max_fetches} fetches)"}
            )

        self.visited_urls.add(canonical)
        self._record_domain_fetch(url)
        self.budget.record_fetch()

        try:
            response = await fetch_pages(
                urls=[url], client=self.tinyfish_client
            )

            if response.errors:
                err = response.errors[0]
                return json.dumps({"error": err.error, "url": url})

            if not response.results:
                return json.dumps({"error": "No content returned", "url": url})

            result = response.results[0]
            return json.dumps(
                {
                    "title": result.title,
                    "url": result.url,
                    "text": result.text,
                    "outbound_links": [
                        {"url": l.url, "anchor_text": l.anchor_text}
                        for l in result.outbound_links
                    ],
                }
            )
        except Exception as e:
            logger.error("Fetch failed", url=url, error=str(e))
            return json.dumps({"error": f"Fetch failed: {str(e)}"})

    def _execute_scratchpad(
        self, args: dict[str, Any], scratchpad: Optional[Any]
    ) -> str:
        if scratchpad is None:
            return json.dumps({"error": "Scratchpad not available"})

        action = args.get("action", "")
        key = args.get("key", "")
        content = args.get("content", "")
        source_url = args.get("source_url", "")

        try:
            if action == "write":
                scratchpad.run("write", key=key, content=content, source_url=source_url)
                next_hint = ""
                if key == "plan":
                    next_hint = "NEXT STEP: Write a checklist to scratchpad (key: 'checklist') with sub-questions to answer."
                elif key == "checklist":
                    next_hint = "NEXT STEP: Search for the first checklist item using the search tool."
                elif key.startswith("facts_"):
                    next_hint = "NEXT STEP: Continue with next checklist item or call finish if done."
                return json.dumps({"status": "written", "key": key, "next_step": next_hint})
            elif action == "append":
                scratchpad.run("append", key=key, content=content, source_url=source_url)
                return json.dumps({"status": "appended", "key": key, "next_step": "Continue with next step."})
            elif action == "read":
                value = scratchpad.run("read", key=key)
                return json.dumps({"key": key, "content": value})
            elif action == "list":
                keys = scratchpad.run("list")
                return json.dumps({"keys": keys, "next_step": "Continue with next step."})
            elif action == "delete":
                scratchpad.run("delete", key=key)
                return json.dumps({"status": "deleted", "key": key, "next_step": "Continue with next step."})
            else:
                return json.dumps({"error": f"Unknown scratchpad action: {action}"})
        except Exception as e:
            return json.dumps({"error": f"Scratchpad error: {str(e)}"})

    def get_budget_status(self) -> dict[str, Any]:
        return {
            "steps_used": self.budget.steps_used,
            "steps_remaining": self.budget.max_steps - self.budget.steps_used,
            "fetches_used": self.budget.fetches_used,
            "fetches_remaining": self.budget.max_fetches - self.budget.fetches_used,
            "searches_used": self.budget.search_calls_used,
            "searches_remaining": self.budget.max_search_calls - self.budget.search_calls_used,
            "tokens_used": self.budget.tokens_used,
            "tokens_remaining": self.budget.max_tokens_total - self.budget.tokens_used,
            "time_elapsed": time.time() - self.budget.start_time,
            "time_remaining": self.budget.wall_clock_seconds - (time.time() - self.budget.start_time),
            "stalled": self.budget.stalled(),
            "visited_urls": len(self.visited_urls),
            "seen_queries": len(self.seen_queries),
        }


def create_tool_registry(
    config: Config,
    tinyfish_client: Optional[TinyFishClient] = None,
) -> ToolRegistry:
    return ToolRegistry(
        agent_config=config.agent,
        crawl_config=config.crawl,
        tinyfish_client=tinyfish_client,
    )
