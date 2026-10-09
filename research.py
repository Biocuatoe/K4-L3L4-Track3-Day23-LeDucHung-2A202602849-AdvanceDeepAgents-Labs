"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to the validator
MAX_SLUG = 60
MODEL_RETRIES = 10       # parallel subagents easily exceed a provider's per-minute quota (429) or hit 503: retry patiently
RECURSION_LIMIT = 1000   # graph steps of the LEAD only; subagents are bounded by their own middleware (agents.py)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[\W_]+", "-", str(topic or "").lower()).strip("-")[:MAX_SLUG].strip("-")
    return slug or "topic"


def harden_model(model):
    """Raise the provider client's retry count (model.py is not ours to edit); no-op for models without `max_retries`."""
    current = getattr(model, "max_retries", None)
    if isinstance(current, int) and current < MODEL_RETRIES:
        model.max_retries = MODEL_RETRIES
    return model


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        "Produce a well-structured English survey report on this topic, following your instructions and "
        "REPORT_TEMPLATE.md: plan with write_todos, split the topic into at least 3 independent sub-questions, delegate "
        "them to `researcher` subagents in parallel, merge their notes into sources.json, write the report body "
        "(without a References section), run the finalizer and then the validator until it prints OK, and finish "
        "with a citation spot-check."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Counted from the lead's messages only: subagent tool calls and tokens are not included, so the token figure
    undercounts the real cost (the subagents are usually the larger part).
    """
    calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += int(usage.get("input_tokens") or 0)
        tokens["output"] += int(usage.get("output_tokens") or 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls.get("task", 0),
            "tool_calls": dict(calls), "tokens": tokens}


def _write_atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    Raises RuntimeError, having written NOTHING, when the report is missing/empty or sources.json is missing/invalid.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    if not raw_sources:
        raise RuntimeError(f"the agent produced no sources at {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources.decode("utf-8"))
        report.decode("utf-8")
    except ValueError as exc:  # includes UnicodeDecodeError
        raise RuntimeError(f"report or sources.json is not valid UTF-8 / JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError("sources.json must be a non-empty JSON list of objects")
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(s["source"]) for s in sources if s.get("source")})}
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    path = reports_dir / f"{slug}.md"
    _write_atomic(reports_dir / f"{slug}.sources.json", raw_sources)
    _write_atomic(reports_dir / f"{slug}.meta.json", json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
    _write_atomic(path, report)
    return path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = harden_model(make_model())
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
        start = time.monotonic()
        with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
            path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, str(model_name))
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - e.g. GraphRecursionError, provider or sandbox errors
        print(f"FAILED: {type(exc).__name__}: {exc}"[:1000], file=sys.stderr)
        return 1
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
