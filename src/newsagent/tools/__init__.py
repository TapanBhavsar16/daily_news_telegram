from __future__ import annotations

from newsagent.tools.telegram import (
    TelegramClient,
    TelegramMessage,
    SendResult,
    escape_html,
    format_story_message,
    send_telegram_message,
    create_telegram_client,
)
from newsagent.tools.tinyfish import (
    TinyFishClient,
    SearchResult,
    SearchResponse,
    FetchResult,
    FetchResponse,
    Link,
    search_web,
    fetch_pages,
    create_tinyfish_client,
)
from newsagent.tools.registry import (
    ToolRegistry,
    Budget,
    DomainPolicy,
    create_tool_registry,
)

__all__ = [
    "TelegramClient",
    "TelegramMessage",
    "SendResult",
    "escape_html",
    "format_story_message",
    "send_telegram_message",
    "create_telegram_client",
    "TinyFishClient",
    "SearchResult",
    "SearchResponse",
    "FetchResult",
    "FetchResponse",
    "Link",
    "search_web",
    "fetch_pages",
    "create_tinyfish_client",
    "ToolRegistry",
    "Budget",
    "DomainPolicy",
    "create_tool_registry",
]