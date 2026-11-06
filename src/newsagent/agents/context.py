from __future__ import annotations

from typing import Any


def trim_old_tool_results(
    messages: list[dict[str, Any]], keep_last: int = 6
) -> list[dict[str, Any]]:
    if not messages:
        return messages

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    other_messages = messages[1:] if system_msg else messages

    tool_result_indices = [
        i
        for i, m in enumerate(other_messages)
        if m.get("role") == "tool"
    ]

    if len(tool_result_indices) <= keep_last:
        return messages

    cutoff_idx = tool_result_indices[-keep_last]
    new_other = []
    for i, msg in enumerate(other_messages):
        if i < cutoff_idx and msg.get("role") == "tool":
            new_other.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": "[cleared; important facts are in the scratchpad]",
                }
            )
        else:
            new_other.append(msg)

    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(new_other)
    return result


def render_state(
    scratchpad: Any,
    budget: Any,
    max_chars: int = 12000,
) -> str:
    parts = []

    import time

    elapsed = budget.elapsed_seconds() if hasattr(budget, "elapsed_seconds") else max(
        0.0, time.time() - getattr(budget, "start_time", time.time())
    )
    budget_info = {
        "steps_used": budget.steps_used,
        "steps_remaining": max(0, budget.max_steps - budget.steps_used),
        "fetches_used": budget.fetches_used,
        "fetches_remaining": max(0, budget.max_fetches - budget.fetches_used),
        "searches_used": budget.search_calls_used,
        "searches_remaining": max(0, budget.max_search_calls - budget.search_calls_used),
        "time_elapsed": elapsed,
        "time_remaining": max(0, budget.wall_clock_seconds - elapsed),
        "stalled": budget.stalled(budget.stall_steps),
    }
    parts.append(
        f"BUDGET: Steps {budget_info['steps_used']}/{budget_info['steps_remaining'] + budget_info['steps_used']}, "
        f"Fetches {budget_info['fetches_used']}/{budget_info['fetches_remaining'] + budget_info['fetches_used']}, "
        f"Searches {budget_info['searches_used']}/{budget_info['searches_remaining'] + budget_info['searches_used']}, "
        f"Time {budget_info['time_elapsed']:.0f}/{budget_info['time_remaining'] + budget_info['time_elapsed']:.0f}s"
    )

    if budget_info["stalled"]:
        parts.append("WARNING: No new facts added in recent steps. Consider finishing.")

    notes = scratchpad.render(limit_chars=max_chars)
    if notes:
        parts.append("\nSCRATCHPAD (your persistent notes):")
        parts.append(notes)
    else:
        parts.append("\nSCRATCHPAD: (empty)")

    return "\n".join(parts)


def force_finish(
    messages: list[dict[str, Any]],
    scratchpad: Any,
    llm_client: Any,
    tools: list[Any],
) -> dict[str, Any]:
    system_prompt = (
        "Your research budget is exhausted. You MUST answer now using ONLY your scratchpad notes. "
        "Do not call any more tools. Output a JSON object with the finish tool:\n"
        '{"thought": "budget exhausted, answering from notes", "tool": "finish", '
        '"args": {"answer": "...", "confidence": 0.0-1.0, "open_gaps": [...]}}'
    )

    messages.append({"role": "system", "content": system_prompt})

    response = llm_client.chat(
        messages=messages,
        tools=tools,
        temperature=0.0,
    )

    if response.tool_calls:
        for call in response.tool_calls:
            if call.name == "finish":
                return call.arguments

    return {
        "answer": response.content or "Unable to complete research within budget.",
        "confidence": 0.3,
        "open_gaps": ["Budget exhausted before completion"],
    }
