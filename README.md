# NewsAgent - Daily News to Telegram

News-to-Telegram agent with deep research capabilities. Fetches news on a topic, researches each story, and delivers a curated digest to Telegram.

## Quick Start

```bash
# 1. Clone and enter the repo
cd daily_news_telegram

# 2. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys (see Configuration below)

# 4. Run the agent
newsagent run "AI chips" --freshness 3
```

## Prerequisites

- Python 3.11+
- TinyFish API key (for web search) - get from [tinyfish.io](https://tinyfish.io)
- Telegram Bot Token - create via [@BotFather](https://t.me/BotFather)
- (Optional) NVIDIA NIM API key or OpenAI API key for LLM models

## Installation

### Standard Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Development Install (with pre-commit hooks)
```bash
pip install -e ".[dev]"
pre-commit install
```

## Configuration

### 1. Environment Variables (`.env`)

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
| Variable | Description | Source |
|----------|-------------|--------|
| `TINYFISH_API_KEY` | Web search API key | [tinyfish.io](https://tinyfish.io) |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Target chat/group/channel ID | See [Telegram Setup](TELEGRAM_SETUP.md) |

Optional variables:
| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | For NVIDIA NIM models (default) |
| `OPENAI_API_KEY` | Fallback for OpenAI models |

### 2. Agent Configuration (`config.yaml`)

The main configuration file controls agent behavior:

```yaml
llm:
  model: "nvidia_nim/mistralai/mistral-7b-instruct-v0.3"
  small_model: "nvidia_nim/mistralai/mistral-7b-instruct-v0.3"

agent:
  max_steps: 20
  max_fetches: 8
  max_search_calls: 6
  wall_clock_seconds: 180

news:
  items_per_topic: 3
  candidates_to_research: 5
  max_parallel_agents: 3

telegram:
  parse_mode: "HTML"
  disable_web_page_preview: true
```

Key settings:
- **freshness**: How many days back to search (CLI flag `--freshness`)
- **max_parallel_agents**: Concurrent research agents (reduce if rate limited)
- **model**: LLM model via LiteLLM format (see [LiteLLM docs](https://docs.litellm.ai/docs/providers))

## Usage

### Run Full News Pipeline
```bash
# Basic usage
newsagent run "AI chips"

# With options
newsagent run "renewable energy" --freshness 7 --verbose

# Resume a previous run
newsagent run "AI chips" --run-id abc123
```

### Run Research Agent (Generic)
```bash
newsagent research "What are the latest developments in quantum computing?"
```

### CLI Options

| Command | Description |
|---------|-------------|
| `newsagent run <topic>` | Full news pipeline: discover → research → curate → send to Telegram |
| `newsagent research <question>` | Generic research agent for any question |
| `--config, -c` | Path to config.yaml (default: `config.yaml`) |
| `--verbose, -v` | Verbose output |
| `--freshness, -f` | Days back to search (default: 7) |
| `--run-id` | Resume previous run by ID |

## Telegram Setup

See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for detailed instructions on:
- Creating a bot with @BotFather
- Getting your chat ID (private, group, or channel)
- Testing message delivery
- Rate limits and troubleshooting

## Project Structure

```
src/newsagent/
├── agents/          # Research and crawler agents
├── graph/           # LangGraph pipeline (discovery → research → curation)
├── tools/           # TinyFish search, Telegram sender
├── memory/          # SQLite persistence
├── config.py        # Configuration loading
├── logging.py       # Structured logging
└── main.py          # CLI entry point (newsagent command)

tests/               # Unit and integration tests
config.yaml          # Agent configuration
.env.example         # Environment variable template
```

## Development

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=newsagent

# Specific test file
pytest tests/test_crawler.py -v
```

### Code Quality
```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src/newsagent
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Deployment

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -e .
COPY config.yaml .env ./
CMD ["newsagent", "run", "your-topic"]
```

### Systemd Service Example
```ini
[Unit]
Description=Daily News Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/newsagent
ExecStart=/opt/newsagent/.venv/bin/newsagent run "AI chips"
Restart=on-failure
EnvironmentFile=/opt/newsagent/.env

[Install]
WantedBy=multi-user.target
```

### Cron Job (Daily at 8 AM)
```bash
0 8 * * * cd /opt/newsagent && .venv/bin/newsagent run "your-topic" --freshness 1 >> /var/log/newsagent.log 2>&1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| `401 Unauthorized` (Telegram) | Check `TELEGRAM_BOT_TOKEN` in `.env` |
| `chat not found` (Telegram) | Verify `TELEGRAM_CHAT_ID` and press Start in bot |
| Rate limited (TinyFish) | Reduce `max_parallel_agents` in config.yaml |
| Model errors | Check API keys; try fallback with `OPENAI_API_KEY` |

## License

MIT License - see LICENSE file for details.