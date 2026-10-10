from __future__ import annotations

import pytest

from newsagent.agents.crawler import (
    Crawler,
    Frontier,
    FrontierEntry,
    create_crawler,
)
from newsagent.config import CrawlConfig
from newsagent.tools.tinyfish import Link


class TestFrontier:
    def setup_method(self):
        config = CrawlConfig(
            max_depth=2,
            max_pages=8,
            max_pages_per_domain=3,
            blocked_domains=["blocked.com"],
            allowed_domains=[],
        )
        self.frontier = Frontier(config)

    def test_add_seed_urls(self):
        urls = [
            ("https://example.com/page1", "Page 1"),
            ("https://example.com/page2", "Page 2"),
        ]
        self.frontier.add_seed_urls(urls)

        assert len(self.frontier._heap) == 2
        entry = self.frontier.pop_best()
        assert entry is not None
        assert entry.url in ["https://example.com/page1", "https://example.com/page2"]
        assert entry.depth == 0

    def test_add_seed_urls_dedupe(self):
        urls = [
            ("https://example.com/page1", "Page 1"),
            ("https://example.com/page1", "Page 1 Again"),
        ]
        self.frontier.add_seed_urls(urls)

        assert len(self.frontier._heap) == 1

    def test_add_seed_urls_blocked_domain(self):
        config = CrawlConfig(blocked_domains=["blocked.com"])
        frontier = Frontier(config)

        urls = [
            ("https://blocked.com/page1", "Blocked"),
            ("https://example.com/page2", "Allowed"),
        ]
        frontier.add_seed_urls(urls)

        assert len(frontier._heap) == 1
        entry = frontier.pop_best()
        assert entry.url == "https://example.com/page2"

    def test_add_seed_urls_allowed_domains(self):
        config = CrawlConfig(allowed_domains=["allowed.com"])
        frontier = Frontier(config)

        urls = [
            ("https://allowed.com/page1", "Allowed"),
            ("https://example.com/page2", "Not Allowed"),
        ]
        frontier.add_seed_urls(urls)

        assert len(frontier._heap) == 1
        entry = frontier.pop_best()
        assert entry.url == "https://allowed.com/page1"

    def test_add_links(self):
        links = [
            Link(url="https://example.com/link1", anchor_text="Link 1"),
            Link(url="https://example.com/link2", anchor_text="Link 2"),
        ]
        self.frontier.add_links(links, "https://example.com/source", current_depth=0)

        assert len(self.frontier._heap) == 2
        entry = self.frontier.pop_best()
        assert entry.depth == 1
        assert entry.source_url == "https://example.com/source"

    def test_add_links_depth_limit(self):
        config = CrawlConfig(max_depth=1)
        frontier = Frontier(config)

        links = [Link(url="https://example.com/link1", anchor_text="Link 1")]
        frontier.add_links(links, "https://example.com/source", current_depth=1)

        assert len(frontier._heap) == 0

    def test_add_links_domain_limit(self):
        config = CrawlConfig(max_pages_per_domain=1)
        frontier = Frontier(config)

        links = [
            Link(url="https://example.com/link1", anchor_text="Link 1"),
            Link(url="https://example.com/link2", anchor_text="Link 2"),
        ]
        frontier.add_links(links, "https://example.com/source", current_depth=0)
        entry1 = frontier.pop_best()
        frontier.mark_visited(entry1.url)

        frontier.add_links(
            [Link(url="https://example.com/link3", anchor_text="Link 3")],
            "https://example.com/source",
            current_depth=0,
        )

        assert len(frontier._heap) == 1

    def test_pop_best_marks_visited(self):
        urls = [("https://example.com/page1", "Page 1")]
        self.frontier.add_seed_urls(urls)

        entry = self.frontier.pop_best()
        assert entry is not None
        self.frontier.mark_visited(entry.url)

        entry2 = self.frontier.pop_best()
        assert entry2 is None

    def test_canonical_url_dedupe(self):
        urls = [
            ("https://example.com/page1", "Page 1"),
            ("https://example.com/page1/", "Page 1 Trailing Slash"),
            ("https://example.com/page1?query=1", "Page 1 Query"),
        ]
        self.frontier.add_seed_urls(urls)

        assert len(self.frontier._heap) == 1

    def test_score_primary_source_bonus(self):
        primary_url = "https://company.com/newsroom/press-release-2024"
        secondary_url = "https://news-aggregator.com/article/123"

        primary_score = self.frontier._score_link(primary_url, "Press Release", "", 0)
        secondary_score = self.frontier._score_link(secondary_url, "Article", "", 0)

        assert primary_score > secondary_score

    def test_score_keyword_bonus(self):
        keyword_url = "https://example.com/announced-new-product-launch"
        generic_url = "https://example.com/page-123"

        keyword_score = self.frontier._score_link(keyword_url, "New Product Launch", "", 0)
        generic_score = self.frontier._score_link(generic_url, "Page", "", 0)

        assert keyword_score > generic_score

    def test_score_navigation_penalty(self):
        nav_url = "https://example.com/login?return=/dashboard"
        content_url = "https://example.com/article/content"

        nav_score = self.frontier._score_link(nav_url, "Login", "", 0)
        content_score = self.frontier._score_link(content_url, "Article", "", 0)

        assert content_score > nav_score

    def test_score_social_penalty(self):
        social_url = "https://twitter.com/user/status/123"
        content_url = "https://example.com/article"

        social_score = self.frontier._score_link(social_url, "Tweet", "", 0)
        content_score = self.frontier._score_link(content_url, "Article", "", 0)

        assert content_score > social_score

    def test_score_recency_bonus(self):
        recent_url = "https://example.com/2024/10/15/article"
        old_url = "https://example.com/article-old"

        recent_score = self.frontier._score_link(recent_url, "Article", "", 0)
        old_score = self.frontier._score_link(old_url, "Article", "", 0)

        assert recent_score > old_score

    def test_score_depth_penalty(self):
        shallow_score = self.frontier._score_link("https://example.com/page", "Link", "", 0)
        deep_score = self.frontier._score_link("https://example.com/page", "Link", "", 2)

        assert shallow_score > deep_score

    def test_get_top_entries(self):
        urls = [
            ("https://example.com/low", "Low Priority"),
            ("https://example.com/high", "Breaking News Announced"),
        ]
        self.frontier.add_seed_urls(urls)

        top = self.frontier.get_top_entries(2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_render_frontier_state(self):
        urls = [("https://example.com/page1", "Page 1")]
        self.frontier.add_seed_urls(urls)

        rendered = self.frontier.render_frontier_state()
        assert "Frontier" in rendered
        assert "example.com" in rendered

    def test_get_stats(self):
        urls = [("https://example.com/page1", "Page 1")]
        self.frontier.add_seed_urls(urls)
        entry = self.frontier.pop_best()
        self.frontier.mark_visited(entry.url)

        stats = self.frontier.get_stats()
        assert stats["total_pages"] == 1
        assert stats["visited_count"] == 1
        assert "example.com" in stats["domain_counts"]


class TestCrawler:
    def setup_method(self):
        config = CrawlConfig(
            max_depth=2,
            max_pages=8,
            max_pages_per_domain=3,
        )
        self.crawler = create_crawler(config)

    def test_initialize_from_search_results(self):
        search_results = [
            {"url": "https://example.com/result1", "title": "Result 1"},
            {"url": "https://example.com/result2", "title": "Result 2"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        assert len(self.crawler.frontier._heap) == 2

    def test_get_next_url(self):
        search_results = [
            {"url": "https://example.com/result1", "title": "Result 1"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        url, depth = self.crawler.get_next_url()
        assert url == "https://example.com/result1"
        assert depth == 0

    def test_process_fetch_result(self):
        search_results = [
            {"url": "https://example.com/result1", "title": "Result 1"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        url, depth = self.crawler.get_next_url()
        links = [
            Link(url="https://example.com/link1", anchor_text="Link 1"),
            Link(url="https://example.com/link2", anchor_text="Link 2"),
        ]

        result = self.crawler.process_fetch_result(
            url=url,
            depth=depth,
            title="Page Title",
            text="Page content",
            outbound_links=links,
        )

        assert result.url == url
        assert result.depth == depth
        assert result.title == "Page Title"
        assert len(result.outbound_links) == 2

        stats = self.crawler.get_stats()
        assert stats["total_pages"] == 1
        assert stats["frontier_size"] == 2

    def test_crawler_follows_link_to_primary_source(self):
        search_results = [
            {"url": "https://news-site.com/article", "title": "News Article"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        url, depth = self.crawler.get_next_url()
        primary_links = [
            Link(url="https://company.com/newsroom/official-statement", anchor_text="Official Statement"),
        ]

        self.crawler.process_fetch_result(
            url=url,
            depth=depth,
            title="News Article",
            text="Article content",
            outbound_links=primary_links,
        )

        next_url, next_depth = self.crawler.get_next_url()
        assert next_url == "https://company.com/newsroom/official-statement"
        assert next_depth == 1

    def test_get_frontier_state(self):
        search_results = [
            {"url": "https://example.com/result1", "title": "Result 1"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        state = self.crawler.get_frontier_state()
        assert "Frontier" in state
        assert "example.com" in state

    def test_get_stats(self):
        search_results = [
            {"url": "https://example.com/result1", "title": "Result 1"},
        ]
        self.crawler.initialize_from_search_results(search_results)

        url, depth = self.crawler.get_next_url()
        self.crawler.process_fetch_result(
            url=url,
            depth=depth,
            title="Title",
            text="Content",
            outbound_links=[],
        )

        stats = self.crawler.get_stats()
        assert stats["total_pages"] == 1
        assert stats["visited_count"] == 1


class TestCrawlerIntegration:
    def test_full_crawl_flow_follows_primary_source_link(self):
        config = CrawlConfig(
            max_depth=2,
            max_pages=5,
            max_pages_per_domain=3,
        )
        crawler = create_crawler(config)

        search_results = [
            {
                "url": "https://tech-news.com/ai-chip-announcement",
                "title": "New AI Chip Announced",
            },
        ]
        crawler.initialize_from_search_results(search_results)

        url1, depth1 = crawler.get_next_url()
        assert url1 == "https://tech-news.com/ai-chip-announcement"
        assert depth1 == 0

        outbound_links = [
            Link(url="https://nvidia.com/newsroom/ai-chip-launch", anchor_text="Official Launch Page"),
            Link(url="https://twitter.com/nvidia/status/123", anchor_text="Tweet"),
            Link(url="https://tech-news.com/related-article", anchor_text="Related"),
        ]

        crawler.process_fetch_result(
            url=url1,
            depth=depth1,
            title="New AI Chip Announced",
            text="NVIDIA announced a new AI chip...",
            outbound_links=outbound_links,
        )

        url2, depth2 = crawler.get_next_url()
        assert url2 == "https://nvidia.com/newsroom/ai-chip-launch"
        assert depth2 == 1

        crawler.process_fetch_result(
            url=url2,
            depth=depth2,
            title="NVIDIA Launches New AI Chip",
            text="Official press release from NVIDIA...",
            outbound_links=[],
        )

        stats = crawler.get_stats()
        assert stats["total_pages"] == 2
        assert stats["max_depth_reached"] == 1
        assert "nvidia.com" in stats["domain_counts"]
        assert "tech-news.com" in stats["domain_counts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])