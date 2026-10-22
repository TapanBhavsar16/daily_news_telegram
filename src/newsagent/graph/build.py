from __future__ import annotations

import uuid
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from newsagent.config import Config, load_config
from newsagent.graph.nodes import (
    backfill_node,
    collect_node,
    compose_node,
    dedupe_story_node,
    discover_node,
    fan_out_node,
    filter_seen_node,
    plan_queries_node,
    post_telegram_node,
    rank_select_node,
    research_agent_node,
    save_memory_node,
    should_backfill,
    should_research,
    should_widen_queries,
    shortlist_node,
    verify_recency_node,
    widen_queries_node,
)
from newsagent.graph.state import TopicState
from newsagent.memory.store import MemoryStore
from newsagent.logging import get_logger

logger = get_logger(__name__)

def create_initial_state(
    topic: str,
    config: Config,
    freshness_days: int | None = None,
    run_id: str | None = None,
) -> TopicState:
    news_cfg = config.news
    return TopicState(
        topic=topic,
        freshness_days=freshness_days or 7,
        round=0,
        queries=[],
        candidates=[],
        research=[],
        selected=[],
        message="",
        sent=False,
        lead_notes={},
        errors=[],
        run_id=run_id or str(uuid.uuid4())[:12],
        max_discovery_rounds=news_cfg.max_discovery_rounds,
        candidates_to_research=news_cfg.candidates_to_research,
        items_per_topic=news_cfg.items_per_topic,
        max_parallel_agents=news_cfg.max_parallel_agents,
        max_backfill_rounds=news_cfg.max_backfill_rounds,
        min_items_to_post=news_cfg.min_items_to_post,
    )


def build_graph(config: Config | None = None, checkpointer: AsyncSqliteSaver | None = None):
    config = config or load_config()

    graph = StateGraph(TopicState)

    graph.add_node("plan_queries", plan_queries_node)
    graph.add_node("discover", discover_node)
    graph.add_node("filter_seen", filter_seen_node)
    graph.add_node("shortlist", shortlist_node)
    graph.add_node("widen_queries", widen_queries_node)
    graph.add_node("fan_out", fan_out_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("collect", collect_node)
    graph.add_node("verify_recency", verify_recency_node)
    graph.add_node("rank_select", rank_select_node)
    graph.add_node("dedupe_story", dedupe_story_node)
    graph.add_node("backfill", backfill_node)
    graph.add_node("compose", compose_node)
    graph.add_node("post_telegram", post_telegram_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "discover")
    graph.add_edge("discover", "filter_seen")
    graph.add_edge("filter_seen", "shortlist")
    graph.add_conditional_edges("shortlist", should_widen_queries, {
        "widen_queries": "widen_queries",
        "shortlist": "fan_out",
    })
    graph.add_edge("widen_queries", "discover")
    graph.add_conditional_edges("fan_out", should_research, {
        "research_agent": "research_agent",
        "collect": "collect",
    })
    graph.add_edge("research_agent", "collect")
    graph.add_edge("collect", "verify_recency")
    graph.add_edge("verify_recency", "rank_select")
    graph.add_edge("rank_select", "dedupe_story")
    graph.add_conditional_edges("dedupe_story", should_backfill, {
        "backfill": "backfill",
        "compose": "compose",
    })
    graph.add_edge("backfill", "fan_out")
    graph.add_edge("compose", "post_telegram")
    graph.add_edge("post_telegram", "save_memory")
    graph.add_edge("save_memory", END)

    if checkpointer is None:
        db_path = config.database.path
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = aiosqlite.connect(db_path)
        checkpointer = AsyncSqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


def _aggregate_costs(result: dict[str, Any]) -> dict[str, Any]:
    """Aggregate costs from all research agents in the pipeline result."""
    total = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        "steps_used": 0,
        "fetches_used": 0,
        "searches_used": 0,
        "wall_time_seconds": 0.0,
        "agents_count": 0,
    }
    
    for r in result.get("research", []):
        total["total_tokens"] += r.get("tokens_used", 0)
        total["steps_used"] += r.get("steps_used", 0)
        total["fetches_used"] += r.get("fetches_used", 0)
        total["searches_used"] += r.get("searches_used", 0)
        total["wall_time_seconds"] += r.get("wall_time_seconds", 0)
        total["agents_count"] += 1
    
    return total


async def run_news_pipeline(
    topic: str,
    config: Config | None = None,
    freshness_days: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    initial_state = create_initial_state(topic, config, freshness_days, run_id)

    graph = build_graph(config)

    memory = MemoryStore(config)
    memory.start_run(initial_state["run_id"], topic)

    try:
        result = await graph.ainvoke(
            initial_state,
            config={
                "recursion_limit": 100,
                "configurable": {
                    "thread_id": initial_state["run_id"],
                    "app_config": config,
                },
            },
        )
        
        cost_summary = _aggregate_costs(result)
        logger.info(
            "News pipeline completed",
            run_id=initial_state["run_id"],
            topic=topic,
            cost_summary=cost_summary,
            stories_selected=len(result.get("selected", [])),
            sent=result.get("sent", False),
        )
        
        result["cost_summary"] = cost_summary
        return result
    except Exception as e:
        memory.finish_run(initial_state["run_id"], "failed", error=str(e))
        raise


__all__ = ["build_graph", "create_initial_state", "run_news_pipeline"]