"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET  # arXiv answers with Atom XML
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRY_STATUS = {429, 500, 502, 503, 504}
HTTP_TIMEOUT = 30.0
SUMMARY_CHARS = 600
FETCH_CHARS = 12000
ARXIV_RETRY = {"attempts": 6, "base": 3.0, "cap": 60.0}   # arXiv answers 429 to a whole shared IP
EXA_RETRY = {"attempts": 6, "base": 5.0, "cap": 60.0}     # Exa's free tier is rate limited quickly
ARXIV_MIN_GAP = 3.0                                       # seconds between two arXiv calls (their API etiquette)


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    A server-provided retry_after wins (capped at `cap`); otherwise the delay is base * 2**attempt plus up to 50% random
    jitter, capped at `cap`. No sleep after the last attempt: the error is re-raised. Any other exception propagates
    immediately (programming and authentication errors must not be retried).
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(float(exc.retry_after), 0.0), cap)
            else:
                delay = min(base * 2 ** attempt, cap)
                delay = min(delay + random.uniform(0, delay * 0.5), cap)
            time.sleep(delay)


def _request(method, url, **kwargs):
    """One HTTP call. Transient failures become RetryableError; other HTTP errors raise httpx.HTTPStatusError."""
    try:
        response = httpx.request(method, url, timeout=HTTP_TIMEOUT, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRY_STATUS:
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        raise RetryableError(f"HTTP {response.status_code} from {url.split('?')[0]}", retry_after=retry_after)
    response.raise_for_status()
    return response


def _clean(text, limit=None):
    text = " ".join(str(text or "").split())
    return text[:limit] if limit else text


def _clamp(value, low, high):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


def _error(exc, secrets=()):
    """'ERROR: ...' string for the agent, with every secret (and any exaApiKey=... URL parameter) removed."""
    message = f"ERROR: {type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>").replace(quote(secret, safe=""), "<redacted>")
    return re.sub(r"(?i)(exaApiKey=)[^&\s'\")]+", r"\1<redacted>", message)[:500]


# ---- arXiv ----
ATOM = "{http://www.w3.org/2005/Atom}"
_arxiv_lock = threading.Lock()
_arxiv_last_call = [None]


def _arxiv_throttle():
    """Hold the lock while waiting, so parallel subagents are spaced at least ARXIV_MIN_GAP seconds apart."""
    with _arxiv_lock:
        last = _arxiv_last_call[0]
        if last is not None:
            wait = ARXIV_MIN_GAP - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        _arxiv_last_call[0] = time.monotonic()


def _arxiv_terms(query):
    return re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query or ""))


def _parse_arxiv(xml_text):
    records = []
    for entry in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        raw_id = _clean(entry.findtext(f"{ATOM}id"))
        if "/abs/" not in raw_id:  # arXiv reports API errors as a pseudo-entry
            continue
        paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
        records.append({
            "id": paper_id,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "published": _clean(entry.findtext(f"{ATOM}published"))[:10],
            "title": _clean(entry.findtext(f"{ATOM}title")),
            "summary": _clean(entry.findtext(f"{ATOM}summary"), SUMMARY_CHARS),
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords (every keyword must match), newest first. Use a few short keywords, not a sentence.
    Returns a JSON list of {id, url, published, title, summary}; url is https://arxiv.org/abs/<id>."""
    terms = _arxiv_terms(query)
    if not terms:
        return "NO RESULTS"
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": _clamp(max_results, 1, 30), "start": 0}

    def call():
        _arxiv_throttle()
        return _request("GET", ARXIV_URL, params=params)

    try:
        records = _parse_arxiv(with_retry(call, **ARXIV_RETRY).text)
    except Exception as exc:  # noqa: BLE001 - tools never raise
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- Hugging Face ----
def _hf_records(items, prefer_ai_summary=False):
    if not isinstance(items, list):
        raise ValueError(f"unexpected response of type {type(items).__name__}")
    records = []
    for item in items:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        paper_id = str(paper["id"])
        summary = (paper.get("ai_summary") if prefer_ai_summary else None) or paper.get("summary") or item.get("summary")
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _clean(paper.get("publishedAt") or item.get("publishedAt"))[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(summary, SUMMARY_CHARS),
            "upvotes": paper["upvotes"] if isinstance(paper.get("upvotes"), int) else 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper["githubStars"] if isinstance(paper.get("githubStars"), int) else 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": _clamp(limit, 1, 100)}
    if (date or "").strip():
        params["date"] = date.strip()
    try:
        records = _hf_records(with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json())
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    needle = (keyword or "").strip().lower()
    if needle:
        records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
    records.sort(key=lambda r: r["upvotes"], reverse=True)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    if not (query or "").strip():
        return "NO RESULTS"
    params = {"q": query.strip(), "limit": _clamp(limit, 1, 50)}
    try:
        records = _hf_records(with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params)).json(),
                              prefer_ai_summary=True)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- web search / fetch through the Exa MCP endpoint ----
_RATE_TEXT = re.compile(r"rate.?limit|too many requests|exceeded.{0,40}(limit|quota)", re.I)
_RATE_META = re.compile(r"rate.?limit|throttl", re.I)


def _exa_endpoint():
    key = os.getenv("EXA_API_KEY", "").strip()
    return (f"{EXA_URL}?exaApiKey={quote(key, safe='')}" if key else EXA_URL), key


def _sse_messages(body):
    """Yield the JSON payload of every server-sent event (or of a plain JSON body)."""
    if body.lstrip().startswith("{"):
        yield json.loads(body)
        return
    event = []
    for line in body.splitlines() + [""]:
        if line.startswith("data:"):
            event.append(line[5:].lstrip())
        elif not line.strip() and event:
            data, event = "\n".join(event), []
            try:
                yield json.loads(data)
            except ValueError:
                continue


def _rate_limited(result, text):
    """Exa's free tier answers HTTP 200 with a flag in result._meta and/or a short explanatory text."""
    meta = result.get("_meta")
    if isinstance(meta, dict) and any(v and _RATE_META.search(str(k)) for k, v in meta.items()):
        return True
    return (len(text) < 800 or bool(result.get("isError"))) and bool(_RATE_TEXT.search(text))


def _exa_call(tool_name, arguments):
    """Return (text, None) or (None, "ERROR: ..."); rate limits are retried and never returned as content."""
    url, key = _exa_endpoint()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def call():
        response = _request("POST", url, json=payload, headers=headers)
        message = next((m for m in _sse_messages(response.text) if isinstance(m, dict) and ("result" in m or "error" in m)),
                       None)
        if message is None:
            raise ValueError("no JSON-RPC message in the Exa response")
        if message.get("error"):
            err = message["error"]
            text = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if _RATE_TEXT.search(text):
                raise RetryableError(f"Exa rate limit: {text}")
            raise ValueError(f"Exa JSON-RPC error: {text}")
        result = message["result"] if isinstance(message.get("result"), dict) else {}
        text = "\n".join(p.get("text", "") for p in result.get("content", [])
                         if isinstance(p, dict) and p.get("type") == "text")
        if _rate_limited(result, text):
            raise RetryableError("Exa rate limit (HTTP 200)")
        if result.get("isError"):
            raise ValueError(f"Exa tool error: {text[:300]}")
        return text

    try:
        return with_retry(call, **EXA_RETRY), None
    except Exception as exc:  # noqa: BLE001
        return None, _error(exc, secrets=(key,))


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs.
    Each result has 'Title:', 'URL:' and 'Published:' lines. Use it for blogs, project pages, surveys and anything
    that is not on arXiv or Hugging Face."""
    if not (query or "").strip():
        return "NO RESULTS"
    arguments = {"query": query.strip(), "objective": (objective or "").strip() or f"Find pages about: {query.strip()}",
                 "numResults": _clamp(num_results, 1, 10)}
    text, error = _exa_call("web_search_exa", arguments)
    return error or text.strip() or "NO RESULTS"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    if not (url or "").strip():
        return "NO RESULTS"
    text, error = _exa_call("web_fetch_exa", {"urls": [url.strip()]})
    if error:
        return error
    return text.strip()[:FETCH_CHARS] or "NO RESULTS"


# ---- registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
