from __future__ import annotations

from newsagent.agents.scratchpad import Scratchpad, ScratchpadNote, create_scratchpad
from newsagent.agents.research_agent import ResearchAgent, ResearchResult, research
from newsagent.agents.context import trim_old_tool_results, render_state, force_finish

__all__ = [
    "Scratchpad",
    "ScratchpadNote",
    "create_scratchpad",
    "ResearchAgent",
    "ResearchResult",
    "research",
    "trim_old_tool_results",
    "render_state",
    "force_finish",
]