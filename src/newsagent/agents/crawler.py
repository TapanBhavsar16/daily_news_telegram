from __future__ import annotations

import heapq
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from newsagent.config import CrawlConfig
from newsagent.logging import get_logger
from newsagent.tools.tinyfish import Link

logger = get_logger(__name__)


PRIMARY_SOURCE_PATTERNS = [
    r"investor\.relations",
    r"newsroom",
    r"press\.releases?",
    r"sec\.gov",
    r"filing",
    r"annualreport",
    r"earnings",
    r"quarterly",
    r"government",
    r"\.gov\b",
    r"\.edu\b",
    r"official",
    r"statement",
    r"regulator",
]

NAVIGATION_PATTERNS = [
    r"login",
    r"signup",
    r"register",
    r"cart",
    r"checkout",
    r"search\?",
    r"tag/",
    r"category/",
    r"author/",
    r"page/\d+",
    r"#",
    r"javascript:",
    r"mailto:",
    r"tel:",
    r"\.pdf$",
    r"\.jpg$",
    r"\.png$",
    r"\.gif$",
    r"\.css$",
    r"\.js$",
]

SOCIAL_PATTERNS = [
    r"twitter\.com",
    r"x\.com",
    r"facebook\.com",
    r"instagram\.com",
    r"linkedin\.com",
    r"youtube\.com",
    r"tiktok\.com",
    r"reddit\.com",
    r"pinterest\.com",
    r"medium\.com",
    r"substack\.com",
]

RECENCY_PATTERNS = [
    r"/202[0-9]/\d{2}/",
    r"/\d{4}-\d{2}-\d{2}/",
    r"\d{8}",
]


@dataclass
class FrontierEntry:
    score: float
    depth: int
    url: str
    anchor_text: str = ""
    source_url: str = ""
    added_at: float = field(default_factory=time.time)

    def __lt__(self, other: FrontierEntry) -> bool:
        return self.score > other.score


@dataclass
class CrawlResult:
    url: str
    depth: int
    title: str
    text: str
    outbound_links: list[Link]
    score: float


class Frontier:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self._heap: list[FrontierEntry] = []
        self._visited: set[str] = set()
        self._in_heap: set[str] = set()
        self._domain_counts: dict[str, int] = {}
        self._urls_by_depth: dict[int, list[str]] = {}
        self._total_pages = 0
        self._max_depth_reached = 0

    def _canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        except Exception:
            return url

    def _get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _is_allowed_domain(self, url: str) -> bool:
        domain = self._get_domain(url)
        if not domain:
            return True

        if self.config.blocked_domains and any(b in domain for b in self.config.blocked_domains):
            return False

        if self.config.allowed_domains and not any(a in domain for a in self.config.allowed_domains):
            return False

        return True

    def _check_domain_limit(self, url: str) -> bool:
        domain = self._get_domain(url)
        if not domain:
            return True
        count = self._domain_counts.get(domain, 0)
        return count < self.config.max_pages_per_domain

    def _record_domain_fetch(self, url: str):
        domain = self._get_domain(url)
        if domain:
            self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1

    def add_seed_urls(self, urls: list[tuple[str, str]], source_url: str = ""):
        for url, anchor in urls:
            canonical = self._canonical_url(url)
            if canonical in self._visited or canonical in self._in_heap:
                continue
            if not self._is_allowed_domain(url):
                continue
            score = self._score_link(url, anchor, source_url, depth=0)
            entry = FrontierEntry(
                score=score,
                depth=0,
                url=url,
                anchor_text=anchor,
                source_url=source_url,
            )
            heapq.heappush(self._heap, entry)
            self._in_heap.add(canonical)
            logger.debug("Added seed URL to frontier", url=url, score=score, anchor=anchor)

    def add_links(self, links: list[Link], source_url: str, current_depth: int):
        next_depth = current_depth + 1
        if next_depth > self.config.max_depth:
            return

        for link in links:
            canonical = self._canonical_url(link.url)
            if canonical in self._visited or canonical in self._in_heap:
                continue
            if not self._is_allowed_domain(link.url):
                continue
            if not self._check_domain_limit(link.url):
                continue

            score = self._score_link(link.url, link.anchor_text, source_url, depth=next_depth)
            entry = FrontierEntry(
                score=score,
                depth=next_depth,
                url=link.url,
                anchor_text=link.anchor_text,
                source_url=source_url,
            )
            heapq.heappush(self._heap, entry)
            self._in_heap.add(canonical)
            if next_depth > self._max_depth_reached:
                self._max_depth_reached = next_depth
            logger.debug("Added link to frontier", url=link.url, score=score, depth=next_depth)

    def _score_link(self, url: str, anchor_text: str, source_url: str, depth: int) -> float:
        score = 100.0

        score -= depth * 15

        anchor_lower = anchor_text.lower()
        url_lower = url.lower()

        story_keywords = ["announced", "launch", "release", "acquired", "merger", "funding", "ipo", "earnings", "report", "study", "research"]
        keyword_matches = sum(1 for kw in story_keywords if kw in anchor_lower or kw in url_lower)
        score += keyword_matches * 10

        primary_matches = sum(1 for pattern in PRIMARY_SOURCE_PATTERNS if re.search(pattern, url_lower, re.IGNORECASE))
        score += primary_matches * 25

        nav_penalty = sum(1 for pattern in NAVIGATION_PATTERNS if re.search(pattern, url_lower))
        score -= nav_penalty * 20

        social_penalty = sum(1 for pattern in SOCIAL_PATTERNS if re.search(pattern, url_lower))
        score -= social_penalty * 30

        recency_bonus = sum(1 for pattern in RECENCY_PATTERNS if re.search(pattern, url_lower))
        score += recency_bonus * 10

        try:
            source_domain = urlparse(source_url).netloc if source_url else ""
            link_domain = urlparse(url).netloc
            if source_domain and link_domain != source_domain:
                score += 5
        except Exception:
            pass

        return max(score, 0.1)

    def get_top_entries(self, n: int = 10) -> list[FrontierEntry]:
        sorted_entries = sorted(self._heap, key=lambda e: e.score, reverse=True)
        return sorted_entries[:n]

    def pop_best(self) -> FrontierEntry | None:
        while self._heap:
            entry = heapq.heappop(self._heap)
            canonical = self._canonical_url(entry.url)
            if canonical not in self._visited:
                return entry
        return None

    def mark_visited(self, url: str):
        canonical = self._canonical_url(url)
        self._visited.add(canonical)
        self._in_heap.discard(canonical)
        self._record_domain_fetch(url)
        self._total_pages += 1

        depth = 0
        for entry in self._heap:
            if self._canonical_url(entry.url) == canonical:
                depth = entry.depth
                break
        if depth not in self._urls_by_depth:
            self._urls_by_depth[depth] = []
        self._urls_by_depth[depth].append(canonical)

    def can_continue(self) -> bool:
        if self._total_pages >= self.config.max_pages:
            return False
        if not self._heap:
            return False
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_pages": self._total_pages,
            "frontier_size": len(self._heap),
            "visited_count": len(self._visited),
            "domain_counts": dict(self._domain_counts),
            "urls_by_depth": {k: len(v) for k, v in self._urls_by_depth.items()},
            "max_depth_reached": self._max_depth_reached,
        }

    def render_frontier_state(self, limit: int = 10) -> str:
        entries = self.get_top_entries(limit)
        if not entries:
            return "Frontier is empty."

        lines = ["Frontier (top entries):"]
        for i, entry in enumerate(entries, 1):
            lines.append(
                f"  {i}. [{entry.score:.1f}] depth={entry.depth} {entry.url} "
                f"(anchor: {entry.anchor_text[:60] if entry.anchor_text else 'N/A'})"
            )
        return "\n".join(lines)


class Crawler:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.frontier = Frontier(config)

    def initialize_from_search_results(self, search_results: list[dict[str, Any]]):
        seed_urls = []
        for r in search_results:
            url = r.get("url", "")
            title = r.get("title", "")
            if url:
                seed_urls.append((url, title))
        self.frontier.add_seed_urls(seed_urls)

    def get_next_url(self) -> tuple[str, int] | None:
        entry = self.frontier.pop_best()
        if entry is None:
            return None
        return entry.url, entry.depth

    def process_fetch_result(
        self,
        url: str,
        depth: int,
        title: str,
        text: str,
        outbound_links: list[Link],
    ) -> CrawlResult:
        self.frontier.mark_visited(url)
        self.frontier.add_links(outbound_links, url, depth)
        return CrawlResult(
            url=url,
            depth=depth,
            title=title,
            text=text,
            outbound_links=outbound_links,
            score=0.0,
        )

    def get_frontier_state(self) -> str:
        return self.frontier.render_frontier_state()

    def get_stats(self) -> dict[str, Any]:
        return self.frontier.get_stats()


def create_crawler(config: CrawlConfig) -> Crawler:
    return Crawler(config)