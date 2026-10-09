"""Offline tests for research.py and agents.py (no LLM, no sandbox, no network)."""
import json
from types import SimpleNamespace

import pytest

import agents
import research
from research import save_outputs, slugify, summarize

REPORT_BODY = b"# T\n\n## TL;DR\n- x [1]\n\n## References\n[1] A. arxiv. https://arxiv.org/abs/1 (2025-01-01)\n"
SOURCES = [{"n": 1, "id": "1", "url": "https://arxiv.org/abs/1", "title": "A", "date": "2025-01-01", "source": "arxiv"},
           {"n": 2, "id": "2", "url": "https://huggingface.co/papers/2", "title": "B", "date": "", "source": "hf-search"},
           {"n": 3, "id": "3", "url": "https://x.org", "title": "C", "date": "", "source": "arxiv"}]


class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p), error=None) for p in paths]


def msg(tool_calls=(), usage=None):
    return SimpleNamespace(tool_calls=[{"name": n} for n in tool_calls], usage_metadata=usage)


# ---------- slugify ----------
@pytest.mark.parametrize("topic, expected", [
    ("survey about world model", "survey-about-world-model"),
    ("  LLM Agents & Tool-Use!! ", "llm-agents-tool-use"),
    ("", "topic"), ("   ", "topic"), ("!!!???", "topic"), (None, "topic"),
    ("../../etc/passwd", "etc-passwd"), ("..\\..\\x", "x"), ("a/b\\c", "a-b-c"),
    ("mô hình thế giới", "mô-hình-thế-giới"),
])
def test_slugify(topic, expected):
    assert slugify(topic) == expected


def test_slugify_limits_length_without_trailing_hyphen():
    slug = slugify("word " * 40)
    assert len(slug) <= 60 and not slug.endswith("-") and not slug.startswith("-")
    assert slugify("a" * 59 + " b") == "a" * 59


# ---------- harden_model ----------
def test_harden_model_raises_low_retry_counts_only():
    low, high, none = SimpleNamespace(max_retries=2), SimpleNamespace(max_retries=50), SimpleNamespace()
    assert research.harden_model(low).max_retries == research.MODEL_RETRIES
    assert research.harden_model(high).max_retries == 50
    assert not hasattr(research.harden_model(none), "max_retries")


# ---------- summarize ----------
def test_summarize_counts_real_messages_only():
    messages = [msg(["write_todos", "task", "task", "task"], {"input_tokens": 100, "output_tokens": 10}),
                msg(["execute"], {"input_tokens": 200, "output_tokens": 20}), msg(), SimpleNamespace(content="plain")]
    out = summarize(messages, 12.349, "m")
    assert out == {"model": "m", "elapsed_s": 12.3, "subagent_calls": 3,
                   "tool_calls": {"write_todos": 1, "task": 3, "execute": 1}, "tokens": {"input": 300, "output": 30}}


def test_summarize_empty():
    assert summarize([], 0, "m")["subagent_calls"] == 0


# ---------- save_outputs ----------
def files(report=REPORT_BODY, sources=None):
    return {agents.REPORT_PATH: report, agents.SOURCES_PATH: json.dumps(SOURCES if sources is None else sources).encode()}


def test_save_outputs_writes_three_files(tmp_path):
    messages = [msg(["task"] * 3, {"input_tokens": 5, "output_tokens": 6})]
    path = save_outputs(FakeBackend(files()), "Survey About X", messages, 3.0, "model-x", reports_dir=tmp_path / "reports")
    assert path == tmp_path / "reports" / "survey-about-x.md" and path.read_bytes() == REPORT_BODY
    meta = json.loads((tmp_path / "reports" / "survey-about-x.meta.json").read_text(encoding="utf-8"))
    assert meta["topic"] == "Survey About X" and meta["n_sources"] == 3 and meta["subagent_calls"] == 3
    assert meta["source_families"] == ["arxiv", "hf-search"] and meta["tokens"] == {"input": 5, "output": 6}
    assert json.loads((tmp_path / "reports" / "survey-about-x.sources.json").read_text(encoding="utf-8")) == SOURCES
    assert sorted(p.name for p in (tmp_path / "reports").iterdir()) == [
        "survey-about-x.md", "survey-about-x.meta.json", "survey-about-x.sources.json"]  # no .tmp leftovers


@pytest.mark.parametrize("bad", [
    {},                                                         # nothing produced
    {agents.SOURCES_PATH: b"[]"},                               # no report
    {agents.REPORT_PATH: b"  \n", agents.SOURCES_PATH: b"[{}]"},  # blank report
    {agents.REPORT_PATH: REPORT_BODY},                          # no sources
    {agents.REPORT_PATH: REPORT_BODY, agents.SOURCES_PATH: b"{not json"},
    {agents.REPORT_PATH: REPORT_BODY, agents.SOURCES_PATH: b"[]"},
    {agents.REPORT_PATH: REPORT_BODY, agents.SOURCES_PATH: b'{"a": 1}'},
    {agents.REPORT_PATH: REPORT_BODY, agents.SOURCES_PATH: b"[1, 2]"},
    {agents.REPORT_PATH: b"\xff\xfe", agents.SOURCES_PATH: b'[{"n": 1}]'},
])
def test_save_outputs_failure_writes_nothing(tmp_path, bad):
    reports = tmp_path / "reports"
    with pytest.raises(RuntimeError):
        save_outputs(FakeBackend(bad), "topic x", [], 1.0, "m", reports_dir=reports)
    assert not reports.exists() or list(reports.iterdir()) == []


# ---------- main ----------
def test_main_without_topic_returns_2(capsys):
    assert research.main("  ") == 2
    assert "usage" in capsys.readouterr().err


def test_main_cleans_up_sandbox_when_agent_fails(monkeypatch, capsys):
    events = []

    class Box:
        def execute(self, command):
            events.append(("execute", command))

        def upload_files(self, items):
            events.append(("upload", sorted(p for p, _ in items)))

    class Ctx:
        def __enter__(self):
            return Box()

        def __exit__(self, *exc):
            events.append(("closed",))
            return False

    class Agent:
        def invoke(self, payload, config):
            assert config["recursion_limit"] == research.RECURSION_LIMIT
            raise RuntimeError("model exploded")

    monkeypatch.setattr(research, "make_model", lambda: SimpleNamespace(model_name="fake", max_retries=2))
    monkeypatch.setattr(research, "open_sandbox", lambda: Ctx())
    monkeypatch.setattr(research, "build_lead_agent", lambda backend, model: Agent())
    assert research.main("topic") == 1
    assert events[-1] == ("closed",)
    assert ("upload", sorted([agents.VALIDATOR_PATH, agents.FINALIZER_PATH])) in events
    assert "FAILED: model exploded" in capsys.readouterr().err


# ---------- agents ----------
def test_subagent_specs():
    by_name = {s["name"]: s for s in agents.build_subagents()}
    assert set(by_name) == {"researcher", "citation-checker"}
    assert [t.name for t in by_name["citation-checker"]["tools"]] == ["web_fetch"]
    assert {t.name for t in by_name["researcher"]["tools"]} == {
        "arxiv_search", "hf_daily_papers", "hf_search_papers", "web_search", "web_fetch"}
    for spec in by_name.values():
        assert spec["description"] and spec["system_prompt"] and spec["middleware"]
        kinds = {type(m).__name__: m for m in spec["middleware"]}
        assert kinds["ModelCallLimitMiddleware"].run_limit == agents.SUB_MODEL_CALLS
        assert kinds["ToolCallLimitMiddleware"].run_limit == agents.SUB_TOOL_CALLS


def test_prompts_carry_the_workspace_paths_and_rules():
    for path in (agents.NOTES_DIR, agents.SOURCES_PATH, agents.REPORT_PATH, agents.VALIDATOR_PATH, agents.FINALIZER_PATH):
        assert path in agents.LEAD_PROMPT
    assert "write_todos" in agents.LEAD_PROMPT and "task" in agents.LEAD_PROMPT and "## References" in agents.LEAD_PROMPT
    assert "UNTRUSTED" in agents.RESEARCHER_PROMPT and "UNTRUSTED" in agents.LEAD_PROMPT.upper()
    for verdict in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNVERIFIABLE"):
        assert verdict in agents.CHECKER_PROMPT


def test_lead_agent_builds_with_fake_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    agent = agents.build_lead_agent(backend=None, model=Fake(messages=iter([])))
    assert hasattr(agent, "invoke")
