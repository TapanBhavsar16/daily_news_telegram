from __future__ import annotations

import tempfile
import pytest

from newsagent.agents.scratchpad import Scratchpad
from newsagent.config import ScratchpadConfig


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    import os
    os.unlink(db_path)


@pytest.fixture
def scratchpad_config():
    return ScratchpadConfig(
        max_note_chars=1000,
        max_total_chars=5000,
        render_limit_chars=2000,
    )


@pytest.fixture
def pad(scratchpad_config, temp_db):
    return Scratchpad(
        run_id="test-run",
        agent_id="test-agent",
        config=scratchpad_config,
        db_path=temp_db,
    )


def test_write_and_read(pad):
    pad.run("write", key="plan", content="Research AI news")
    content = pad.run("read", key="plan")
    assert content == "Research AI news"


def test_append(pad):
    pad.run("write", key="notes", content="First note")
    pad.run("append", key="notes", content="Second note")
    content = pad.run("read", key="notes")
    assert "First note" in content
    assert "Second note" in content


def test_list_keys(pad):
    pad.run("write", key="plan", content="Plan")
    pad.run("write", key="facts_1", content="Fact 1", source_url="https://example.com")
    keys = pad.run("list")
    assert "plan" in keys
    assert "facts_1" in keys


def test_delete(pad):
    pad.run("write", key="temp", content="Temp")
    pad.run("delete", key="temp")
    keys = pad.run("list")
    assert "temp" not in keys


def test_fact_note_requires_source_url(pad):
    with pytest.raises(ValueError, match="requires at least one source_url"):
        pad.run("write", key="facts_1", content="Fact without source")


def test_fact_note_with_source_url(pad):
    pad.run("write", key="facts_1", content="Fact with source", source_url="https://example.com")
    content = pad.run("read", key="facts_1")
    assert content == "Fact with source"


def test_note_size_limit(pad):
    long_content = "x" * 1500  # Exceeds max_note_chars=1000
    with pytest.raises(ValueError, match="exceeds max_note_chars"):
        pad.run("write", key="big", content=long_content)


def test_total_size_limit(pad):
    pad.run("write", key="note1", content="x" * 900)
    pad.run("write", key="note2", content="y" * 900)
    pad.run("write", key="note3", content="z" * 900)
    pad.run("write", key="note4", content="w" * 900)
    pad.run("write", key="note5", content="v" * 900)
    with pytest.raises(ValueError, match="exceed max_total_chars"):
        pad.run("write", key="note6", content="u" * 900)


def test_append_exceeds_note_limit(pad):
    pad.run("write", key="note", content="x" * 800)
    with pytest.raises(ValueError, match="exceed max_note_chars"):
        pad.run("append", key="note", content="y" * 500)


def test_append_exceeds_total_limit(pad):
    pad.run("write", key="note1", content="x" * 400)
    pad.run("write", key="note2", content="y" * 400)
    pad.run("write", key="note3", content="z" * 400)
    pad.run("write", key="note4", content="w" * 400)
    pad.run("write", key="note5", content="v" * 400)
    # note5 is 400, total = 2000, append 3200 would exceed total 5000 but note limit is checked first
    # Since 400 + 3200 = 3600 > 1000, it hits note limit first
    with pytest.raises(ValueError, match="exceed max_note_chars"):
        pad.run("append", key="note5", content="u" * 3200)


def test_render_includes_reserved_keys_first(pad):
    pad.run("write", key="plan", content="My plan")
    pad.run("write", key="checklist", content="Item 1\nItem 2")
    pad.run("write", key="facts_1", content="Fact 1", source_url="https://ex.com")
    pad.run("write", key="facts_2", content="Fact 2", source_url="https://ex.com")
    pad.run("write", key="conflicts", content="Conflict found")
    pad.run("write", key="open_questions", content="Question?")

    rendered = pad.render()
    assert "## plan" in rendered
    assert "## checklist" in rendered
    assert "## conflicts" in rendered
    assert "## open_questions" in rendered
    # Reserved keys should appear before facts
    plan_pos = rendered.index("## plan")
    facts_pos = rendered.index("## facts_1")
    assert plan_pos < facts_pos


def test_render_truncates_at_limit(pad):
    pad.run("write", key="plan", content="Plan")
    for i in range(10):
        pad.run("write", key=f"facts_{i}", content=f"Fact {i}", source_url="https://ex.com")

    rendered = pad.render(limit_chars=500)
    assert len(rendered) <= 600  # Some buffer for formatting


def test_facts_method_returns_only_facts(pad):
    pad.run("write", key="plan", content="Plan")
    pad.run("write", key="facts_1", content="Fact 1", source_url="https://ex.com")
    pad.run("write", key="facts_2", content="Fact 2", source_url="https://ex.com")
    pad.run("write", key="checklist", content="Item")

    facts = pad.facts()
    assert len(facts) == 2
    assert all(f["key"].startswith("facts_") for f in facts)


def test_persistence_across_instances(pad, temp_db, scratchpad_config):
    pad.run("write", key="persistent", content="Survives restart", source_url="https://ex.com")

    # Create new instance with same run_id and agent_id
    pad2 = Scratchpad(
        run_id="test-run",
        agent_id="test-agent",
        config=scratchpad_config,
        db_path=temp_db,
    )

    content = pad2.run("read", key="persistent")
    assert content == "Survives restart"


def test_agent_isolation(pad, temp_db, scratchpad_config):
    pad.run("write", key="agent1_note", content="Agent 1 data")

    pad2 = Scratchpad(
        run_id="test-run",
        agent_id="different-agent",
        config=scratchpad_config,
        db_path=temp_db,
    )

    pad2.run("write", key="agent2_note", content="Agent 2 data")

    assert "agent1_note" in pad.run("list")
    assert "agent2_note" not in pad.run("list")
    assert "agent2_note" in pad2.run("list")
    assert "agent1_note" not in pad2.run("list")


def test_clear(pad):
    pad.run("write", key="note1", content="Note 1")
    pad.run("write", key="note2", content="Note 2")
    pad.clear()
    assert pad.run("list") == []


def test_get_note(pad):
    pad.run("write", key="test", content="Content", source_url="https://ex.com")
    # Manually set source_date for testing
    if not pad._db_initialized:
        pad._load_from_db()
    pad._notes["test"].source_date = "2024-01-01"
    pad._save_to_db("test")

    note = pad.get_note("test")
    assert note is not None
    assert note.content == "Content"
    assert note.source_urls == ["https://ex.com"]
    assert note.source_date == "2024-01-01"


def test_unknown_action_raises(pad):
    with pytest.raises(ValueError, match="Unknown action"):
        pad.run("invalid_action", key="test")


def test_read_nonexistent_returns_empty(pad):
    result = pad.run("read", key="nonexistent")
    assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])