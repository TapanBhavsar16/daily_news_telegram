from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from newsagent.config import Config, TelegramConfig, get_settings, get_telegram_bot_token, get_telegram_chat_id
from newsagent.logging import get_logger

logger = get_logger(__name__)

TELEGRAM_MAX_LENGTH = 4096


@dataclass
class TelegramMessage:
    text: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True


@dataclass
class SendResult:
    success: bool
    message_id: Optional[int] = None
    error: Optional[str] = None


class TelegramClient:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        timeout: float = 30.0,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.disable_web_page_preview = disable_web_page_preview
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._client = httpx.AsyncClient(timeout=timeout)

    @retry(
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(3),
    )
    async def send_message(self, message: TelegramMessage) -> SendResult:
        if len(message.text) > TELEGRAM_MAX_LENGTH:
            return await self._send_long_message(message)

        payload = {
            "chat_id": self.chat_id,
            "text": message.text,
            "parse_mode": message.parse_mode,
            "disable_web_page_preview": message.disable_web_page_preview,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/sendMessage", json=payload
            )
            response.raise_for_status()
            data = response.json()

            if data.get("ok"):
                message_id = data.get("result", {}).get("message_id")
                logger.info("Telegram message sent", message_id=message_id, length=len(message.text))
                return SendResult(success=True, message_id=message_id)
            else:
                error = data.get("description", "Unknown error")
                logger.error("Telegram API error", error=error)
                return SendResult(success=False, error=error)

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error("Telegram HTTP error", error=error_msg)
            return SendResult(success=False, error=error_msg)
        except Exception as e:
            logger.error("Telegram send failed", error=str(e))
            return SendResult(success=False, error=str(e))

    async def _send_long_message(self, message: TelegramMessage) -> SendResult:
        parts = self._split_message(message.text)
        last_result = None

        for i, part in enumerate(parts):
            part_msg = TelegramMessage(
                text=part,
                parse_mode=message.parse_mode,
                disable_web_page_preview=message.disable_web_page_preview,
            )
            result = await self.send_message(part_msg)
            last_result = result
            if not result.success:
                logger.error("Failed to send part", part=i + 1, total=len(parts), error=result.error)
                return result

        return last_result or SendResult(success=False, error="No parts sent")

    def _split_message(self, text: str) -> list[str]:
        if len(text) <= TELEGRAM_MAX_LENGTH:
            return [text]

        parts = []
        current = ""

        lines = text.split("\n")
        for line in lines:
            if len(current) + len(line) + 1 > TELEGRAM_MAX_LENGTH:
                if current:
                    parts.append(current)
                current = line
            else:
                current += ("\n" if current else "") + line

        if current:
            parts.append(current)

        return parts

    async def close(self):
        await self._client.aclose()


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def format_story_message(
    topic: str,
    date_str: str,
    stories: list[dict[str, Any]],
    max_length: int = TELEGRAM_MAX_LENGTH,
) -> str:
    header = f"<b>📰 {escape_html(topic)}: top {len(stories)} ({escape_html(date_str)})</b>\n\n"
    parts = [header]

    for i, story in enumerate(stories, 1):
        headline = escape_html(story.get("headline", "Untitled"))
        summary = escape_html(story.get("summary", ""))
        key_details = story.get("key_details", [])
        why_matters = escape_html(story.get("why_it_matters", ""))
        sources = story.get("sources", [])
        pub_date = escape_html(story.get("date", ""))

        story_text = f"{i}. <b>{headline}</b>\n"
        if summary:
            story_text += f"{summary}\n"
        for detail in key_details:
            story_text += f"• {escape_html(detail)}\n"
        if why_matters:
            story_text += f"<i>Why it matters:</i> {why_matters}\n"
        if sources:
            source_links = " · ".join(
                f'<a href="{html.escape(s["url"], quote=True)}">{escape_html(s.get("label", "Source"))}</a>'
                for s in sources
            )
            story_text += f"{source_links}"
            if pub_date:
                story_text += f" · {pub_date}"
            story_text += "\n"
        story_text += "\n"

        if len("\n".join(parts)) + len(story_text) > max_length - 100:
            break
        parts.append(story_text)

    return "".join(parts).strip()


async def send_telegram_message(
    text: str,
    config: Optional[Config] = None,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
) -> SendResult:
    settings = get_settings()
    tg_config = config.telegram if config else None
    bot_token_name = tg_config.bot_token_env if tg_config else "TELEGRAM_BOT_TOKEN"
    chat_id_name = tg_config.chat_id_env if tg_config else "TELEGRAM_CHAT_ID"
    bot_token = getattr(settings, bot_token_name, None)
    chat_id = getattr(settings, chat_id_name, None)
    if not bot_token:
        raise ValueError(f"{bot_token_name} not set in environment")
    if not chat_id:
        raise ValueError(f"{chat_id_name} not set in environment")

    if tg_config:
        parse_mode = tg_config.parse_mode
        disable_web_page_preview = tg_config.disable_web_page_preview

    client = TelegramClient(
        bot_token=bot_token,
        chat_id=chat_id,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
    )

    message = TelegramMessage(
        text=text,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
    )

    try:
        result = await client.send_message(message)
        return result
    finally:
        await client.close()


def create_telegram_client(config: Config) -> TelegramClient:
    settings = get_settings()
    bot_token = get_telegram_bot_token(settings)
    chat_id = get_telegram_chat_id(settings)
    tg_config = config.telegram

    return TelegramClient(
        bot_token=bot_token,
        chat_id=chat_id,
        parse_mode=tg_config.parse_mode,
        disable_web_page_preview=tg_config.disable_web_page_preview,
    )
