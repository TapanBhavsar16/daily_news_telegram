from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _add_lists(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class TopicState(TypedDict):
    topic: str
    freshness_days: int
    round: int
    queries: list[str]
    candidates: list[dict[str, Any]]
    research: Annotated[list[dict[str, Any]], _add_lists]
    selected: list[dict[str, Any]]
    message: str
    sent: bool
    lead_notes: dict[str, Any]
    errors: Annotated[list[str], _add_lists]
    run_id: str
    max_discovery_rounds: int
    candidates_to_research: int
    items_per_topic: int
    max_parallel_agents: int
    max_backfill_rounds: int
    min_items_to_post: int
