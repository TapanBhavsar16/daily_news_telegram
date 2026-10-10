from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from newsagent.config import Config
from newsagent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PostedItem:
    id: str
    topic: str
    title: str
    url: str
    url_hash: str
    title_hash: str
    content_hash: str
    summary: str
    key_facts: list[str]
    entities: list[str]
    posted_at: float
    run_id: str


@dataclass
class PendingPost:
    id: str
    topic: str
    message: str
    items: list[dict[str, Any]]
    created_at: float
    sent_at: float | None
    telegram_message_id: int | None


@dataclass
class RunRecord:
    run_id: str
    topic: str
    started_at: float
    finished_at: float | None
    steps: int
    fetches: int
    searches: int
    tokens: int
    outcome: str
    error: str | None


class MemoryStore:
    def __init__(self, config: Config | None = None, db_path: str | None = None):
        self.config = config
        self.db_path = db_path or (config.database.path if config else "data/newsagent.db")
        self._initialized = False
        self.retention_days = config.database.retention_days if config else 30

    def _init_db(self):
        if self._initialized:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posted_items (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    url_hash TEXT NOT NULL UNIQUE,
                    title_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_facts TEXT NOT NULL,
                    entities TEXT NOT NULL,
                    posted_at REAL NOT NULL,
                    run_id TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_posted_topic ON posted_items (topic)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_posted_posted_at ON posted_items (posted_at)"
            )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_posts (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    message TEXT NOT NULL,
                    items TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    sent_at REAL,
                    telegram_message_id INTEGER
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_topic ON pending_posts (topic)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_created ON pending_posts (created_at)"
            )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    steps INTEGER NOT NULL DEFAULT 0,
                    fetches INTEGER NOT NULL DEFAULT 0,
                    searches INTEGER NOT NULL DEFAULT 0,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT NOT NULL DEFAULT 'in_progress',
                    error TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_topic ON runs (topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_at)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS story_facts (
                    id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    source_url TEXT,
                    FOREIGN KEY (item_id) REFERENCES posted_items (id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_story_facts_item ON story_facts (item_id)")

        self._initialized = True

    @staticmethod
    def _url_hash(url: str) -> str:
        canonical = url.strip().lower()
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    @staticmethod
    def _title_hash(title: str) -> str:
        normalized = " ".join(title.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    @staticmethod
    def _content_hash(summary: str, key_facts: list[str]) -> str:
        content = summary + "|" + "|".join(sorted(key_facts))
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def check_duplicate(
        self, url: str, title: str, summary: str, key_facts: list[str]
    ) -> tuple[bool, str]:
        self._init_db()
        url_h = self._url_hash(url)
        title_h = self._title_hash(title)
        content_h = self._content_hash(summary, key_facts)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT id, url_hash FROM posted_items WHERE url_hash = ?",
                (url_h,),
            ).fetchone()
            if row:
                return True, f"URL already posted (id: {row['id']})"

            row = conn.execute(
                "SELECT id, title_hash FROM posted_items WHERE title_hash = ?",
                (title_h,),
            ).fetchone()
            if row:
                return True, f"Title already posted (id: {row['id']})"

            row = conn.execute(
                "SELECT id, content_hash FROM posted_items WHERE content_hash = ?",
                (content_h,),
            ).fetchone()
            if row:
                return True, f"Content already posted (id: {row['id']})"

        return False, ""

    def add_posted_item(
        self,
        topic: str,
        title: str,
        url: str,
        summary: str,
        key_facts: list[str],
        entities: list[str],
        run_id: str,
    ) -> str:
        self._init_db()
        item_id = str(uuid.uuid4())[:12]
        url_h = self._url_hash(url)
        title_h = self._title_hash(title)
        content_h = self._content_hash(summary, key_facts)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO posted_items
                (id, topic, title, url, url_hash, title_hash, content_hash,
                 summary, key_facts, entities, posted_at, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    topic,
                    title,
                    url,
                    url_h,
                    title_h,
                    content_h,
                    summary,
                    json.dumps(key_facts),
                    json.dumps(entities),
                    time.time(),
                    run_id,
                ),
            )
            fact_id_base = str(uuid.uuid4())[:8]
            for i, fact in enumerate(key_facts):
                conn.execute(
                    "INSERT INTO story_facts (id, item_id, fact, source_url) VALUES (?, ?, ?, ?)",
                    (f"{fact_id_base}_{i}", item_id, fact, ""),
                )

        logger.info("Posted item saved", item_id=item_id, topic=topic, title=title[:50])
        return item_id

    def get_recent_posts(self, topic: str, days: int = 7) -> list[PostedItem]:
        self._init_db()
        cutoff = time.time() - (days * 86400)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM posted_items
                WHERE topic = ? AND posted_at > ?
                ORDER BY posted_at DESC
                """,
                (topic, cutoff),
            ).fetchall()
            return [
                PostedItem(
                    id=row["id"],
                    topic=row["topic"],
                    title=row["title"],
                    url=row["url"],
                    url_hash=row["url_hash"],
                    title_hash=row["title_hash"],
                    content_hash=row["content_hash"],
                    summary=row["summary"],
                    key_facts=json.loads(row["key_facts"]),
                    entities=json.loads(row["entities"]),
                    posted_at=row["posted_at"],
                    run_id=row["run_id"],
                )
                for row in rows
            ]

    def create_pending_post(
        self, topic: str, message: str, items: list[dict[str, Any]]
    ) -> str:
        self._init_db()
        pending_id = str(uuid.uuid4())[:12]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO pending_posts (id, topic, message, items, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (pending_id, topic, message, json.dumps(items), time.time()),
            )
        logger.info("Pending post created", pending_id=pending_id, topic=topic)
        return pending_id

    def mark_pending_sent(
        self, pending_id: str, telegram_message_id: int
    ) -> bool:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE pending_posts
                SET sent_at = ?, telegram_message_id = ?
                WHERE id = ? AND sent_at IS NULL
                """,
                (time.time(), telegram_message_id, pending_id),
            )
            return cursor.rowcount > 0

    def get_unsent_pending(self, topic: str) -> list[PendingPost]:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM pending_posts
                WHERE topic = ? AND sent_at IS NULL
                ORDER BY created_at ASC
                """,
                (topic,),
            ).fetchall()
            return [
                PendingPost(
                    id=row["id"],
                    topic=row["topic"],
                    message=row["message"],
                    items=json.loads(row["items"]),
                    created_at=row["created_at"],
                    sent_at=row["sent_at"],
                    telegram_message_id=row["telegram_message_id"],
                )
                for row in rows
            ]

    def start_run(self, run_id: str, topic: str) -> None:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, topic, started_at, outcome)
                VALUES (?, ?, ?, 'in_progress')
                """,
                (run_id, topic, time.time()),
            )
        logger.info("Run started", run_id=run_id, topic=topic)

    def finish_run(
        self,
        run_id: str,
        outcome: str,
        steps: int = 0,
        fetches: int = 0,
        searches: int = 0,
        tokens: int = 0,
        error: str | None = None,
    ) -> None:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, steps = ?, fetches = ?, searches = ?,
                    tokens = ?, outcome = ?, error = ?
                WHERE run_id = ?
                """,
                (time.time(), steps, fetches, searches, tokens, outcome, error, run_id),
            )
        logger.info("Run finished", run_id=run_id, outcome=outcome, steps=steps)

    def get_run(self, run_id: str) -> RunRecord | None:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None
            return RunRecord(
                run_id=row["run_id"],
                topic=row["topic"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                steps=row["steps"],
                fetches=row["fetches"],
                searches=row["searches"],
                tokens=row["tokens"],
                outcome=row["outcome"],
                error=row["error"],
            )

    def cleanup_old_data(self) -> dict[str, int]:
        self._init_db()
        cutoff = time.time() - (self.retention_days * 86400)
        counts = {"posted_items": 0, "pending_posts": 0, "runs": 0, "story_facts": 0}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM posted_items WHERE posted_at < ?", (cutoff,)
            )
            counts["posted_items"] = cursor.rowcount

            cursor = conn.execute(
                "DELETE FROM pending_posts WHERE created_at < ? AND sent_at IS NOT NULL",
                (cutoff,),
            )
            counts["pending_posts"] = cursor.rowcount

            cursor = conn.execute(
                "DELETE FROM runs WHERE started_at < ? AND finished_at IS NOT NULL",
                (cutoff,),
            )
            counts["runs"] = cursor.rowcount

            cursor = conn.execute(
                "DELETE FROM story_facts WHERE item_id NOT IN (SELECT id FROM posted_items)"
            )
            counts["story_facts"] = cursor.rowcount

        logger.info("Cleanup completed", **counts)
        return counts


def create_memory_store(config: Config) -> MemoryStore:
    return MemoryStore(config=config)
