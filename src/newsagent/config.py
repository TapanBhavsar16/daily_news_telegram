from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    model: str = "nvidia_nim/meta/llama-3.1-70b-instruct"
    small_model: str = "nvidia_nim/meta/llama-3.1-8b-instruct"
    fallbacks: list[str] = Field(default_factory=list)


class ScratchpadConfig(BaseModel):
    max_note_chars: int = 4000
    max_total_chars: int = 30000
    render_limit_chars: int = 12000


class AgentConfig(BaseModel):
    tool_mode: str = "json"
    max_steps: int = 20
    max_fetches: int = 8
    max_search_calls: int = 6
    max_tokens_total: int = 120000
    wall_clock_seconds: int = 180
    stall_steps: int = 3
    trim_keep_last: int = 6
    scratchpad: ScratchpadConfig = Field(default_factory=ScratchpadConfig)


class CrawlConfig(BaseModel):
    max_depth: int = 2
    max_pages: int = 8
    max_pages_per_domain: int = 3
    blocked_domains: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)


class NewsConfig(BaseModel):
    items_per_topic: int = 3
    candidates_to_research: int = 5
    max_discovery_rounds: int = 3
    max_backfill_rounds: int = 1
    max_parallel_agents: int = 3
    min_items_to_post: int = 1


class TinyFishConfig(BaseModel):
    api_key_env: str = "TINYFISH_API_KEY"
    base_url: str = "https://agent.tinyfish.ai"


class TelegramConfig(BaseModel):
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True


class DatabaseConfig(BaseModel):
    path: str = "data/newsagent.db"
    retention_days: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    tinyfish: TinyFishConfig = Field(default_factory=TinyFishConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    TINYFISH_API_KEY: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> Config:
    config_file = Path(config_path) if config_path else Path("config.yaml")
    if config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    return Config()


def get_settings() -> Settings:
    return Settings()


def get_tinyfish_api_key(settings: Settings) -> str:
    api_key = settings.TINYFISH_API_KEY
    if not api_key:
        raise ValueError("TINYFISH_API_KEY not set in environment")
    return api_key


def get_telegram_bot_token(settings: Settings) -> str:
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
    return token


def get_telegram_chat_id(settings: Settings) -> str:
    chat_id = settings.TELEGRAM_CHAT_ID
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID not set in environment")
    return chat_id
