# Telegram Setup

This guide covers only the Telegram side: creating a bot, getting the credentials the news agent needs, and testing delivery. You end up with two values: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

---

## 1. Create the bot

1. Open Telegram and search for **@BotFather** (the official one has a verified badge).
2. Send `/newbot`.
3. Choose a display name (for example `My News Bot`).
4. Choose a username. It must end in `bot` (for example `my_daily_news_bot`).
5. BotFather replies with an **HTTP API token** that looks like `123456789:AAH...`. Copy it.

Put it in your `.env`:
```
TELEGRAM_BOT_TOKEN=123456789:AAH...
```

**Treat the token like a password.** Anyone who has it can control the bot. If it leaks, send `/revoke` to BotFather to get a new one.

---

## 2. Get your chat ID

Pick where the news should go.

### Option A: a private chat with yourself (simplest)

1. Open your new bot in Telegram and press **Start** (or send any message). A bot cannot message you until you do this.
2. In a browser or terminal, open:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
3. Find `"chat":{"id": 123456789, ...}` in the JSON. That number is your chat ID.

If `result` is empty, send the bot another message and reload.

### Option B: a group

1. Create a group and add your bot as a member.
2. Send any message in the group (to be safe, `/start@your_bot_username`).
3. Open the same `getUpdates` URL. Group chat IDs are **negative numbers**, such as `-1001234567890` for supergroups.
4. If the bot does not see messages, you may need to turn off privacy mode: BotFather → `/mybots` → your bot → *Bot Settings* → *Group Privacy* → *Turn off*. This is only needed if you want the bot to read group messages. Sending messages works without it.

### Option C: a channel

1. Create a channel and add your bot as an **administrator** with permission to post messages.
2. Use the channel's `@channelusername` as the chat ID (public channels), or the numeric ID that starts with `-100` (private channels, visible in `getUpdates` after you post something in the channel).

Put the value in your `.env`:
```
TELEGRAM_CHAT_ID=123456789
```

---

## 3. Test delivery

Run this from a terminal (replace the placeholders):

```bash
curl -s -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage" \
  -d chat_id="<YOUR_CHAT_ID>" \
  -d parse_mode="HTML" \
  -d text="<b>Test</b> message from the news agent"
```

A response containing `"ok":true` and a message in your chat means setup is complete.

---

## 4. Message rules the agent follows

- **Length limit:** 4096 characters per message. The agent splits longer digests.
- **HTML mode:** all dynamic text is escaped (`&`, `<`, `>`). Supported tags include `<b>`, `<i>`, `<a href="...">`, and `<code>`.
- **Link previews:** the agent can disable previews (`disable_web_page_preview=true`) to keep digests compact. This is configurable.
- **Rate limits:** roughly 1 message per second to the same chat, and around 20 messages per minute to a group. The agent sends one digest per topic, so this is not an issue in normal use.

---

## 5. Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Wrong or revoked token | Copy the token again from BotFather |
| `400 Bad Request: chat not found` | Wrong chat ID, or you never pressed Start | Press Start in the bot chat, then re-check `getUpdates` |
| `403 Forbidden: bot was blocked by the user` | You blocked the bot | Unblock it in Telegram |
| `403 ... bot is not a member of the channel/group` | Bot not added | Add the bot (as admin for channels) |
| `400 can't parse entities` | Unescaped `<`/`&` in message text | The agent escapes text. If you see this, fall back to plain text mode |
| `getUpdates` returns an empty list | No message sent to the bot yet | Send the bot a message, then reload |
| `409 Conflict` on `getUpdates` | A webhook is set on the bot | Delete it: `https://api.telegram.org/bot<TOKEN>/deleteWebhook` |

---

## 6. Security checklist

- Keep `.env` out of version control (add it to `.gitignore`).
- Use GitHub Actions secrets or your host's secret store if you deploy remotely.
- Use one bot per project so you can revoke it independently.
- Never paste the token in screenshots, logs, or chat messages.
