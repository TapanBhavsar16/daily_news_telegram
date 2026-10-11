from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from newsagent.config import Config, ScratchpadConfig
from newsagent.logging import get_logger

logger = get_logger(__name__)


INJECTION_PATTERNS = [
    r"(?i)SYSTEM\s*:",
    r"(?i)Ignore\s+(?:previous|above|all)?\s*(?:previous|above)?\s*(?:instructions|prompts?)",
    r"(?i)Override\s+(system|previous|all)",
    r"(?i)Forget\s+(everything|all|previous|above)",
    r"(?i)New\s+instructions?",
    r"(?i)Assistant\s*:",
    r"(?i)Human\s*:",
    r"(?i)User\s*:",
    r"(?i)<\|.*?\|>",
    r"(?i)```\s*(system|prompt|instruction)",
]


def sanitize_content(content: str) -> str:
    """Remove potential prompt injection patterns from content."""
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        is_injection = False
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, line.strip()):
                is_injection = True
                logger.warning("Filtered potential injection pattern", line=line[:100])
                break
        if not is_injection:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


@dataclass
class ScratchpadNote:
    key: str
    content: str
    source_urls: list[str] = field(default_factory=list)
    source_date: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


class Scratchpad:
    def __init__(
        self,
        run_id: str,
        agent_id: str,
        config: Optional[ScratchpadConfig] = None,
        db_path: Optional[str] = None,
    ):
        self.run_id = run_id
        self.agent_id = agent_id
        self.config = config or ScratchpadConfig()
        self.db_path = db_path or "data/newsagent.db"
        self._notes: dict[str, ScratchpadNote] = {}
        self._db_initialized = False

    def _init_db(self):
        if self._db_initialized:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scratchpad_notes (
                    run_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_urls TEXT NOT NULL,
                    source_date TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (run_id, agent_id, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scratchpad_run_agent
                ON scratchpad_notes (run_id, agent_id)
            """)
        self._db_initialized = True

    def _load_from_db(self):
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT key, content, source_urls, source_date, updated_at "
                "FROM scratchpad_notes WHERE run_id = ? AND agent_id = ?",
                (self.run_id, self.agent_id),
            ).fetchall()
            for row in rows:
                self._notes[row["key"]] = ScratchpadNote(
                    key=row["key"],
                    content=row["content"],
                    source_urls=json.loads(row["source_urls"]),
                    source_date=row["source_date"],
                    updated_at=row["updated_at"],
                )

    def _save_to_db(self, key: str):
        self._init_db()
        note = self._notes[key]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scratchpad_notes
                (run_id, agent_id, key, content, source_urls, source_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.agent_id,
                    note.key,
                    note.content,
                    json.dumps(note.source_urls),
                    note.source_date,
                    note.updated_at,
                ),
            )

    def _delete_from_db(self, key: str):
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM scratchpad_notes WHERE run_id = ? AND agent_id = ? AND key = ?",
                (self.run_id, self.agent_id, key),
            )

    def _total_chars(self) -> int:
        return sum(len(n.content) for n in self._notes.values())

    def _validate_fact_note(self, key: str, source_urls: list[str]):
        if key.startswith("facts_") and not source_urls:
            raise ValueError(f"Fact note '{key}' requires at least one source_url")

    def run(
        self,
        action: str,
        key: str = "",
        content: str = "",
        source_url: str = "",
    ) -> Any:
        if not self._db_initialized:
            self._load_from_db()

        source_urls = [source_url] if source_url else []

        if action == "write":
            self._validate_fact_note(key, source_urls)
            sanitized_content = sanitize_content(content)
            if len(sanitized_content) > self.config.max_note_chars:
                raise ValueError(f"Note exceeds max_note_chars ({self.config.max_note_chars})")
            if self._total_chars() + len(sanitized_content) > self.config.max_total_chars:
                raise ValueError(f"Total scratchpad size would exceed max_total_chars ({self.config.max_total_chars})")

            self._notes[key] = ScratchpadNote(
                key=key,
                content=sanitized_content,
                source_urls=source_urls,
                updated_at=time.time(),
            )
            self._save_to_db(key)
            return {"status": "written", "key": key}

        elif action == "append":
            if key not in self._notes:
                raise ValueError(f"Note '{key}' does not exist, use 'write' to create")
            existing = self._notes[key]
            sanitized_content = sanitize_content(content)
            new_content = existing.content + "\n" + sanitized_content
            if len(new_content) > self.config.max_note_chars:
                raise ValueError(f"Note would exceed max_note_chars ({self.config.max_note_chars})")
            if self._total_chars() + len(sanitized_content) > self.config.max_total_chars:
                raise ValueError(f"Total scratchpad size would exceed max_total_chars ({self.config.max_total_chars})")

            existing.content = new_content
            existing.source_urls.extend(source_urls)
            existing.updated_at = time.time()
            self._save_to_db(key)
            return {"status": "appended", "key": key}

        elif action == "read":
            if key not in self._notes:
                return ""
            return self._notes[key].content

        elif action == "list":
            return list(self._notes.keys())

        elif action == "delete":
            if key in self._notes:
                del self._notes[key]
                self._delete_from_db(key)
            return {"status": "deleted", "key": key}

        else:
            raise ValueError(f"Unknown action: {action}")

    def render(self, limit_chars: Optional[int] = None) -> str:
        if not self._db_initialized:
            self._load_from_db()

        limit = limit_chars or self.config.render_limit_chars
        reserved_keys = ["plan", "checklist", "conflicts", "open_questions"]
        reserved_present = [k for k in reserved_keys if k in self._notes]
        other_keys = [k for k in self._notes.keys() if k not in reserved_keys]
        other_keys.sort(key=lambda k: self._notes[k].updated_at, reverse=True)

        parts = []
        used_chars = 0

        for k in reserved_present:
            note = self._notes[k]
            entry = f"## {k}\n{note.content}\n"
            if used_chars + len(entry) > limit:
                break
            parts.append(entry)
            used_chars += len(entry)

        for k in other_keys:
            note = self._notes[k]
            entry = f"## {k}\n{note.content}\n"
            if used_chars + len(entry) > limit:
                break
            parts.append(entry)
            used_chars += len(entry)

        remaining = [k for k in other_keys if k not in [p.split("## ")[1].split("\n")[0] for p in parts]]
        if remaining:
            parts.append(f"\n[... {len(remaining)} more notes: {', '.join(remaining[:10])}{'...' if len(remaining) > 10 else ''} ...]")

        return "\n".join(parts)

    def facts(self) -> list[dict[str, Any]]:
        if not self._db_initialized:
            self._load_from_db()
        return [
            {
                "key": note.key,
                "content": note.content,
                "source_urls": note.source_urls,
                "source_date": note.source_date,
                "updated_at": note.updated_at,
            }
            for note in self._notes.values()
            if note.key.startswith("facts_")
        ]

    def get_note(self, key: str) -> Optional[ScratchpadNote]:
        if not self._db_initialized:
            self._load_from_db()
        return self._notes.get(key)

    def clear(self):
        if not self._db_initialized:
            self._load_from_db()
        for key in list(self._notes.keys()):
            self._delete_from_db(key)
        self._notes.clear()


def create_scratchpad(run_id: str, agent_id: str, config: Config) -> Scratchpad:
    return Scratchpad(
        run_id=run_id,
        agent_id=agent_id,
        config=config.agent.scratchpad,
        db_path=config.database.path,
    )