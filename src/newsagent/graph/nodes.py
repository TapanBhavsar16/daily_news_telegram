from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph

from newsagent.agents import research as run_research, ResearchResult
from newsagent.config import Config, NewsConfig
from newsagent.graph.state import TopicState
from newsagent.logging import get_logger
from newsagent.memory.store import MemoryStore
from newsagent.tools.telegram import send_telegram_message
from newsagent.tools.tinyfish import search_web
from langgraph.config import get_config

logger = get_logger(__name__)


def _now_ts() -> float:
    return time.time()


async def plan_queries_node(state: TopicState) -> dict[str, Any]:
    topic = state["topic"]
    freshness_days = state["freshness_days"]
    round_num = state["round"]
    lead_notes = state.get("lead_notes", {})

    recent_titles = lead_notes.get("recent_posted_titles", [])

    system_prompt = f"""You are planning search queries for a news discovery task.
Topic: {topic}
Today's date: {date.today().isoformat()}
Freshness window: {freshness_days} days
Discovery round: {round_num + 1}
Recently posted titles (avoid these): {json.dumps(recent_titles[-10:])}

Write 2-4 specific search queries that will find NEW, RECENT news about this topic.
Queries should be short (2-6 words), specific, and use date-aware terms when helpful.
Avoid generic queries. Think about subtopics, regional angles, and recent developments.

Return ONLY a JSON array of strings. Example:
["AI chip export controls October 2026", "Nvidia H200 Blackwell launch", "AMD MI300X data center adoption"]"""

    from newsagent.llm import create_llm_client

    config: Config = get_config()["configurable"]["app_config"]
    llm = create_llm_client(config.agent, config)

    response = llm.chat(
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.3,
    )

    try:
        queries = json.loads(response.content)
        if not isinstance(queries, list):
            queries = [topic]
    except Exception:
        queries = [topic]

    logger.info("Planned queries", topic=topic, round=round_num, queries=queries)
    return {"queries": queries, "round": round_num + 1}


async def discover_node(state: TopicState) -> dict[str, Any]:
    queries = state["queries"]
    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            search_response = await search_web(query, num_results=10)
            results = getattr(search_response, "results", search_response)
            if isinstance(results, tuple):
                results = results[0]
            for r in results:
                if hasattr(r, "model_dump"):
                    r = r.model_dump()
                elif not isinstance(r, dict):
                    r = {
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "snippet": getattr(r, "snippet", ""),
                    }
                url = r.get("url", "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        except Exception as e:
            logger.warning("Search failed", query=query, error=str(e))

    logger.info("Discovered candidates", count=len(all_results))
    return {"candidates": all_results}


async def filter_seen_node(state: TopicState) -> dict[str, Any]:
    candidates = state["candidates"]
    run_id = state["run_id"]
    topic = state["topic"]

    config: Config = get_config()["configurable"]["app_config"]
    memory = MemoryStore(config)
    memory._init_db()

    recent = memory.get_recent_posts(topic, days=state.get("freshness_days", 7))
    seen_urls = {item.url for item in recent}
    seen_title_hashes = {item.title_hash for item in recent}

    filtered = []
    for c in candidates:
        url = c.get("url", "").strip()
        title = c.get("title", "").strip()

        if not url or not title:
            continue

        from newsagent.memory.store import MemoryStore as MS
        url_hash = MS._url_hash(url)
        title_hash = MS._title_hash(title)

        if url in seen_urls or url_hash in {MS._url_hash(u) for u in seen_urls}:
            continue
        if title_hash in seen_title_hashes:
            continue

        filtered.append(c)

    logger.info("Filtered seen", original=len(candidates), filtered=len(filtered))
    return {"candidates": filtered}


async def shortlist_node(state: TopicState) -> dict[str, Any]:
    candidates = state["candidates"]
    candidates_to_research = state.get("candidates_to_research", 5)

    if not candidates:
        logger.info("No candidates to shortlist")
        return {"candidates": []}

    items_text = "\n".join(
        [f"{i}. {c.get('title', '')} - {c.get('snippet', '')[:150]} ({c.get('url', '')})"
         for i, c in enumerate(candidates[:20])]
    )

    system_prompt = f"""You are selecting the most promising news stories for deep research.
Topic: {state['topic']}
Today's date: {date.today().isoformat()}
Select up to {candidates_to_research} candidates that are:
- Genuinely new (within the last few days)
- Substantive (not just a mention or rumor)
- From credible sources
- Distinct from each other (not the same story from different outlets)

Candidates:
{items_text}

Return ONLY a JSON array of indices (0-based) of the selected candidates. Example: [0, 2, 5]"""

    from newsagent.llm import create_llm_client
    config: Config = get_config()["configurable"]["app_config"]
    llm = create_llm_client(config.agent, config)

    response = llm.chat(
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.2,
    )

    try:
        indices = json.loads(response.content)
        if not isinstance(indices, list):
            indices = list(range(min(candidates_to_research, len(candidates))))
    except Exception:
        indices = list(range(min(candidates_to_research, len(candidates))))

    selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
    logger.info("Shortlisted", count=len(selected))
    return {"candidates": selected}


async def widen_queries_node(state: TopicState) -> dict[str, Any]:
    topic = state["topic"]
    round_num = state["round"]
    lead_notes = state.get("lead_notes", {})

    system_prompt = f"""The previous discovery round found too few viable new stories.
Topic: {topic}
Round: {round_num + 1}
Current lead notes: {json.dumps(lead_notes, default=str)[:500]}

Generate 2-4 WIDER or DIFFERENT search queries to find more stories.
Try: synonyms, subtopics, adjacent angles, "last 24 hours", regional variations, specific companies/people.

Return ONLY a JSON array of strings."""

    from newsagent.llm import create_llm_client
    config: Config = get_config()["configurable"]["app_config"]
    llm = create_llm_client(config.agent, config)

    response = llm.chat(
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.4,
    )

    try:
        queries = json.loads(response.content)
        if not isinstance(queries, list):
            queries = [topic]
    except Exception:
        queries = [topic]

    logger.info("Widened queries", round=round_num, queries=queries)
    return {"queries": queries, "round": round_num + 1}


async def fan_out_node(state: TopicState) -> dict[str, Any]:
    logger.info("Fanning out to research agents", count=len(_build_research_sends(state)))
    return {}


def _build_research_sends(state: TopicState) -> list[Send]:
    candidates = state.get("candidates", [])[: state.get("max_parallel_agents", 3)]
    if not candidates:
        return []
    topic = state["topic"]
    run_id = state["run_id"]
    sends = []
    for index, candidate in enumerate(candidates):
        goal = f"""Research this news story in depth:
Title: {candidate.get('title', '')}
URL: {candidate.get('url', '')}
Snippet: {candidate.get('snippet', '')}
Topic: {topic}
Today: {date.today().isoformat()}

Follow the standard news coverage checklist: confirm what happened, who and when;
include key numbers, seek a primary source and supporting context, explain why it matters.
Return a comprehensive answer with all sources cited."""
        sends.append(Send("research_agent", {
            "goal": goal,
            "candidate": candidate,
            "agent_index": index,
            "run_id": run_id,
            "topic": topic,
        }))
    return sends


def fan_out_dispatch(state: TopicState):
    sends = _build_research_sends(state)
    return sends if sends else "collect"


def should_research(state: TopicState) -> str:
    return "research_agent" if state.get("candidates") else "collect"


async def research_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    goal = state["goal"]
    candidate = state["candidate"]
    agent_index = state["agent_index"]
    run_id = state["run_id"]
    topic = state["topic"]

    agent_id = f"research_{run_id}_{agent_index}"

    logger.info("Starting research agent", agent_id=agent_id, title=candidate.get("title", "")[:50])

    config = state.get("config")
    if config is None:
        config = get_config()["configurable"]["app_config"]

    result: ResearchResult = await run_research(
        goal,
        config=config,
        run_id=run_id,
        agent_id=agent_id,
    )

    research_result = {
        "candidate": candidate,
        "answer": result.answer,
        "confidence": result.confidence,
        "open_gaps": result.open_gaps,
        "sources": result.sources,
        "notes": result.notes,
        "steps_used": result.steps_used,
        "fetches_used": result.fetches_used,
        "searches_used": result.searches_used,
        "tokens_used": result.tokens_used,
        "wall_time_seconds": result.wall_time_seconds,
        "agent_id": agent_id,
        "run_id": run_id,
    }

    return {"research": [research_result]}


async def collect_node(state: TopicState) -> dict[str, Any]:
    return {}


async def verify_recency_node(state: TopicState) -> dict[str, Any]:
    research_results = state.get("research", [])
    freshness_days = state.get("freshness_days", 7)
    cutoff_date = date.today() - timedelta(days=freshness_days)

    verified = []
    for r in research_results:
        confidence = r.get("confidence", 0.0)
        candidate = r.get("candidate", {})

        has_date = any(
            "date" in str(note).lower() or "202" in str(note)
            for note in r.get("notes", {}).values()
        )

        if confidence < 0.3 and not has_date:
            logger.info("Dropping low-confidence undated story", title=candidate.get("title", "")[:50])
            continue

        verified.append(r)

    logger.info("Verified recency", original=len(research_results), verified=len(verified))
    return {"research": verified}


async def rank_select_node(state: TopicState) -> dict[str, Any]:
    research_results = state.get("research", [])
    items_per_topic = state.get("items_per_topic", 3)

    if not research_results:
        logger.info("No research results to rank")
        return {"selected": []}

    items_text = "\n".join(
        [
            f"{i}. Confidence: {r.get('confidence', 0):.2f} | "
            f"Title: {r.get('candidate', {}).get('title', '')[:80]} | "
            f"Sources: {len(r.get('sources', []))} | "
            f"Answer preview: {r.get('answer', '')[:200]}"
            for i, r in enumerate(research_results)
        ]
    )

    system_prompt = f"""Rank these researched stories for a news digest.
Topic: {state['topic']}
Select the top {items_per_topic} stories by:
1. Recency (more recent = better)
2. Importance/impact
3. Source quality (primary sources > aggregators)
4. Agent confidence
5. Completeness (fewer open_gaps = better)

Research results:
{items_text}

Return ONLY a JSON array of indices (0-based) of the top {items_per_topic} stories. Example: [0, 2, 1]"""

    from newsagent.llm import create_llm_client
    config: Config = get_config()["configurable"]["app_config"]
    llm = create_llm_client(config.agent, config)

    response = llm.chat(
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.2,
    )

    try:
        indices = json.loads(response.content)
        if not isinstance(indices, list):
            indices = list(range(min(items_per_topic, len(research_results))))
    except Exception:
        indices = list(range(min(items_per_topic, len(research_results))))

    selected = [
        research_results[i]
        for i in indices
        if isinstance(i, int) and 0 <= i < len(research_results)
    ]
    if not selected:
        selected = research_results[:items_per_topic]
    logger.info("Ranked and selected", count=len(selected))
    return {"selected": selected}


async def dedupe_story_node(state: TopicState) -> dict[str, Any]:
    selected = state.get("selected", [])
    topic = state["topic"]
    run_id = state["run_id"]

    config: Config = get_config()["configurable"]["app_config"]
    memory = MemoryStore(config)
    memory._init_db()

    recent = memory.get_recent_posts(topic, days=30)

    final_selected = []
    for r in selected:
        candidate = r.get("candidate", {})
        answer = r.get("answer", "")
        key_facts = [
            str(value)
            for key, value in r.get("notes", {}).items()
            if key.startswith("facts_")
        ]

        is_dup, reason = memory.check_duplicate(
            url=candidate.get("url", ""),
            title=candidate.get("title", ""),
            summary=answer[:500],
            key_facts=[str(f) for f in key_facts[:5]],
        )

        if is_dup:
            logger.info("Dropping duplicate story", title=candidate.get("title", "")[:50], reason=reason)
            continue

        final_selected.append(r)

    logger.info("Story dedupe complete", original=len(selected), final=len(final_selected))
    return {"selected": final_selected}


async def backfill_node(state: TopicState) -> dict[str, Any]:
    selected = state.get("selected", [])
    research_results = state.get("research", [])
    items_per_topic = state.get("items_per_topic", 3)
    max_backfill = state.get("max_backfill_rounds", 1)
    max_parallel = state.get("max_parallel_agents", 3)

    lead_notes = state.get("lead_notes", {})
    backfill_rounds = lead_notes.get("backfill_rounds", 0)

    if len(selected) >= items_per_topic or backfill_rounds >= max_backfill:
        return {"lead_notes": lead_notes}

    researched_urls = {
        r.get("candidate", {}).get("url", "")
        for r in research_results
    }
    attempted_urls = {
        c.get("url", "") for c in state.get("lead_notes", {}).get("attempted_candidates", [])
    }
    new_candidates = [
        c for c in state.get("candidates", [])
        if c.get("url", "") not in researched_urls | attempted_urls
    ][:max_parallel]

    if not new_candidates:
        return {"lead_notes": {**lead_notes, "backfill_rounds": backfill_rounds + 1}}

    logger.info("Backfilling", need=items_per_topic - len(selected), backfilling=len(new_candidates))

    return {
        "lead_notes": {
            **lead_notes,
            "backfill_rounds": backfill_rounds + 1,
            "attempted_candidates": [
                *lead_notes.get("attempted_candidates", []),
                *new_candidates,
            ],
        },
        "candidates": new_candidates,
        "research": [],
    }


async def compose_node(state: TopicState) -> dict[str, Any]:
    selected = state.get("selected", [])
    topic = state["topic"]

    if not selected:
        return {"message": "", "sent": False}

    from newsagent.tools.telegram import format_story_message

    today_str = date.today().strftime("%d %b %Y")
    
    stories = []
    for r in selected:
        candidate = r.get("candidate", {})
        answer = r.get("answer", "")
        confidence = r.get("confidence", 0.0)
        sources = r.get("sources", [])
        notes = r.get("notes", {})

        title = candidate.get("title", "Untitled")
        
        sentences = [s.strip() for s in answer.split(". ") if s.strip()]
        summary = ". ".join(sentences[:3]) if sentences else answer[:300]
        
        key_facts = [v for k, v in notes.items() if k.startswith("facts_")]
        
        why_matters = sentences[-1] if len(sentences) > 3 else "Significant development in this area."
        
        source_list = []
        for src in sources[:3]:
            source_list.append({"url": src, "label": "Source"})
        
        story = {
            "headline": title,
            "summary": summary,
            "key_details": key_facts[:3],
            "why_it_matters": why_matters,
            "sources": source_list,
            "date": today_str,
        }
        stories.append(story)

    message = format_story_message(topic, today_str, stories)

    if confidence < 0.5:
        message += "\n\n<i>Note: Low confidence (single/limited sources)</i>"

    logger.info("Composed message", length=len(message), items=len(selected))
    return {"message": message, "sent": False}


async def post_telegram_node(state: TopicState) -> dict[str, Any]:
    message = state.get("message", "")
    run_id = state["run_id"]
    topic = state["topic"]
    selected = state.get("selected", [])
    config: Config | None = state.get("config")

    if not message or not selected:
        logger.info("Nothing to post")
        return {"sent": False}

    memory = MemoryStore(config)
    memory._init_db()

    pending_id = memory.create_pending_post(
        topic=topic,
        message=message,
        items=[{"candidate": r["candidate"], "answer": r["answer"], "sources": r["sources"]} for r in selected],
    )

    try:
        result = await send_telegram_message(message, config=config)
        if isinstance(result, dict):
            sent = bool(result.get("success", result.get("ok", False)))
            telegram_msg_id = result.get("message_id") or result.get("result", {}).get("message_id")
            send_error = result.get("error") or result.get("description")
        else:
            sent = bool(getattr(result, "success", False))
            telegram_msg_id = getattr(result, "message_id", None)
            send_error = getattr(result, "error", None)

        if not sent:
            error = send_error or "Telegram did not confirm message delivery"
            logger.error("Telegram send failed", error=error)
            memory.mark_pending_failed(pending_id, error)
            memory.finish_run(run_id, "failed", error=error)
            return {"sent": False, "errors": [f"Telegram send failed: {error}"]}

        if telegram_msg_id is None:
            error = "Telegram returned success without a message ID"
            memory.mark_pending_failed(pending_id, error)
            memory.finish_run(run_id, "failed", error=error)
            return {"sent": False, "errors": [error]}

        memory.mark_pending_sent(pending_id, telegram_msg_id)

        for r in selected:
            candidate = r["candidate"]
            answer = r["answer"]
            key_facts = [v for k, v in r.get("notes", {}).items() if k.startswith("facts_")]
            entities = []  # Could extract entities here

            memory.add_posted_item(
                topic=topic,
                title=candidate.get("title", ""),
                url=candidate.get("url", ""),
                summary=answer[:500],
                key_facts=[str(f) for f in key_facts[:10]],
                entities=entities,
                run_id=run_id,
            )

        memory.finish_run(
            run_id,
            "success",
            steps=len(selected),
            fetches=sum(r.get("fetches_used", 0) for r in selected),
        )
        logger.info("Posted to Telegram", run_id=run_id, message_id=telegram_msg_id)
        return {"sent": True}

    except Exception as e:
        logger.error("Telegram send failed", error=str(e))
        memory.finish_run(run_id, "failed", error=str(e))
        return {"sent": False, "errors": [f"Telegram send failed: {e}"]}


async def save_memory_node(state: TopicState) -> dict[str, Any]:
    return {}


def should_widen_queries(state: TopicState) -> str:
    candidates = state.get("candidates", [])
    round_num = state.get("round", 0)
    max_rounds = state.get("max_discovery_rounds", 3)
    items_per_topic = state.get("items_per_topic", 3)

    viable = len(candidates)
    if viable >= items_per_topic or round_num >= max_rounds:
        return "shortlist"
    return "widen_queries"


def should_backfill(state: TopicState) -> str:
    selected = state.get("selected", [])
    items_per_topic = state.get("items_per_topic", 3)
    max_backfill = state.get("max_backfill_rounds", 1)
    lead_notes = state.get("lead_notes", {})
    backfill_rounds = lead_notes.get("backfill_rounds", 0)

    if len(selected) >= items_per_topic or backfill_rounds >= max_backfill:
        return "compose"
    return "backfill"


def should_research(state: TopicState) -> str:
    candidates = state.get("candidates", [])
    if not candidates:
        return "collect"
    return "research_agent"
