from __future__ import annotations

import asyncio
import sys
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from newsagent.config import Config, load_config
from newsagent.agents import research as run_research, ResearchResult
from newsagent.graph import run_news_pipeline
from newsagent.logging import configure_logging, get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="newsagent",
    help="News-to-Telegram agent with deep research capabilities",
    add_completion=False,
)


@app.command()
def research(
    question: str = typer.Argument(..., help="Research question to answer"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run the generic research agent on a question.

    Example:
        newsagent research "What are the latest developments in AI chips?"
    """
    config = load_config(config_path)
    configure_logging(config.logging.level, config.logging.format)

    if verbose:
        console.print(f"[dim]Config:[/dim] {config.llm.model}, tool_mode={config.agent.tool_mode}")
        console.print(f"[dim]Question:[/dim] {question}")

    console.print(Panel(f"[bold]Researching:[/bold] {question}", expand=False))

    result = asyncio.run(_run_research(question, config))

    _display_result(result)


@app.command()
def run(
    topic: str = typer.Argument(..., help="News topic to research"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    freshness_days: int = typer.Option(7, "--freshness", "-f", help="Days back to search for news"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional run ID for resuming"),
):
    """
    Run the full news pipeline for a topic.

    Example:
        newsagent run "AI chips" --freshness 3
    """
    config = load_config(config_path)
    configure_logging(config.logging.level, config.logging.format)

    if verbose:
        console.print(f"[dim]Config:[/dim] {config.llm.model}, tool_mode={config.agent.tool_mode}")
        console.print(f"[dim]Topic:[/dim] {topic}")
        console.print(f"[dim]Freshness:[/dim] {freshness_days} days")

    console.print(Panel(f"[bold]News Pipeline:[/bold] {topic}", expand=False))

    result = asyncio.run(_run_news_pipeline(topic, config, freshness_days, run_id))

    _display_pipeline_result(result)
    if result.get("errors") or not result.get("sent"):
        raise typer.Exit(code=1)


async def _run_research(question: str, config: Config) -> ResearchResult:
    return await run_research(question, config=config)


async def _run_news_pipeline(
    topic: str, config: Config, freshness_days: int, run_id: Optional[str]
) -> dict[str, Any]:
    return await run_news_pipeline(topic, config, freshness_days, run_id)


def _display_result(result: ResearchResult) -> None:
    console.print("\n[bold green]✓ Research Complete[/bold green]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Answer", result.answer)
    table.add_row("Confidence", f"{result.confidence:.0%}")
    table.add_row("Steps", str(result.steps_used))
    table.add_row("Fetches", str(result.fetches_used))
    table.add_row("Searches", str(result.searches_used))
    table.add_row("Tokens", f"{result.tokens_used:,}")
    table.add_row("Time", f"{result.wall_time_seconds:.1f}s")
    table.add_row("Run ID", result.run_id)

    console.print(table)

    if result.open_gaps:
        console.print("\n[bold yellow]Open Gaps:[/bold yellow]")
        for gap in result.open_gaps:
            console.print(f"  • {gap}")

    if result.sources:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for i, source in enumerate(result.sources, 1):
            console.print(f"  {i}. {source}")

    if result.notes:
        console.print("\n[bold]Scratchpad Notes:[/bold]")
        for key, content in result.notes.items():
            if key.startswith("facts_"):
                console.print(f"  [dim]{key}:[/dim] {content[:200]}..." if len(content) > 200 else f"  [dim]{key}:[/dim] {content}")
            else:
                console.print(f"  [bold]{key}:[/bold] {content[:300]}..." if len(content) > 300 else f"  [bold]{key}:[/bold] {content}")


def _display_pipeline_result(result: dict[str, Any]) -> None:
    console.print("\n[bold green]✓ News Pipeline Complete[/bold green]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Topic", result.get("topic", ""))
    table.add_row("Run ID", result.get("run_id", ""))
    table.add_row("Rounds", str(result.get("round", 0)))
    table.add_row("Candidates Found", str(len(result.get("candidates", []))))
    table.add_row("Stories Researched", str(len(result.get("research", []))))
    table.add_row("Stories Selected", str(len(result.get("selected", []))))
    table.add_row("Message Sent", "Yes" if result.get("sent") else "No")
    table.add_row("Errors", str(len(result.get("errors", []))))

    console.print(table)

    if result.get("errors"):
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result["errors"]:
            console.print(f"  • {err}")

    if result.get("selected"):
        console.print("\n[bold]Selected Stories:[/bold]")
        for i, r in enumerate(result["selected"], 1):
            candidate = r.get("candidate", {})
            title = candidate.get("title", "Untitled")
            confidence = r.get("confidence", 0.0)
            console.print(f"  {i}. {title} (confidence: {confidence:.0%})")

    if result.get("message"):
        console.print("\n[bold]Telegram Message Preview:[/bold]")
        preview = result["message"][:500] + "..." if len(result["message"]) > 500 else result["message"]
        console.print(Markdown(preview))


def main():
    app()


if __name__ == "__main__":
    main()
