"""Offline tests for tools.py: no network, no sleeping (httpx.request, time.sleep and random are mocked)."""
import json

import httpx
import pytest

import tools
from tools import RetryableError, with_retry

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T10:00:00Z</published>
<title>A  Title
 with   breaks</title><summary>  Some
 summary   text. </summary></entry>
<entry><id>http://arxiv.org/api/errors#x</id><title>Error</title><summary>bad</summary></entry>
<entry><id>http://arxiv.org/abs/hep-th/9901001v1</id><published>1999-01-05T00:00:00Z</published><title>Old</title><summary>s</summary></entry>
</feed>"""


def response(status=200, text="", json_body=None, headers=None):
    request = httpx.Request("GET", "https://example.test/x")
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers, request=request)
    return httpx.Response(status, text=text, headers=headers, request=request)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    monkeypatch.setattr(tools.random, "uniform", lambda a, b: b)  # deterministic: maximal jitter
    tools._arxiv_last_call[0] = None
    return sleeps


def patch_http(monkeypatch, *responses):
    """The i-th call gets the i-th response (the last one repeats). Exceptions are raised."""
    calls, queue = [], list(responses)

    def fake(method, url, **kwargs):
        calls.append((method, url, kwargs))
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tools.httpx, "request", fake)
    return calls


# ---------- with_retry ----------
def test_retry_succeeds_after_transient_failure(no_sleep):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise RetryableError("boom")
        return "ok"

    assert with_retry(fn) == "ok"
    assert state["n"] == 3 and len(no_sleep) == 2


def test_retry_exhausted_raises_and_does_not_sleep_after_last(no_sleep):
    def fn():
        raise RetryableError("always")

    with pytest.raises(RetryableError):
        with_retry(fn, attempts=4)
    assert len(no_sleep) == 3  # attempts - 1


def test_retry_exponential_backoff_with_jitter_and_cap(no_sleep):
    def fn():
        raise RetryableError("x")

    with pytest.raises(RetryableError):
        with_retry(fn, attempts=5, base=2.0, cap=10.0)
    # base*2**i = 2,4,8,16 -> + 50% jitter (mocked to the maximum) -> 3,6,12,24 -> capped at 10
    assert no_sleep == [3.0, 6.0, 10.0, 10.0]


def test_retry_after_is_respected_and_capped(no_sleep):
    queue = [RetryableError("x", retry_after=7), RetryableError("x", retry_after=500)]

    def fn():
        if queue:
            raise queue.pop(0)
        return "done"

    assert with_retry(fn, cap=30.0) == "done"
    assert no_sleep == [7.0, 30.0]


def test_non_retryable_exception_is_not_retried(no_sleep):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        with_retry(fn)
    assert state["n"] == 1 and not no_sleep


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_request_marks_transient_statuses_retryable(monkeypatch, status):
    patch_http(monkeypatch, response(status, headers={"Retry-After": "4"}))
    with pytest.raises(RetryableError) as info:
        tools._request("GET", "https://example.test/x")
    assert info.value.retry_after == 4.0


def test_request_transport_error_is_retryable_but_auth_error_is_not(monkeypatch):
    patch_http(monkeypatch, httpx.ConnectTimeout("slow"))
    with pytest.raises(RetryableError):
        tools._request("GET", "https://example.test/x")
    patch_http(monkeypatch, response(401))
    with pytest.raises(httpx.HTTPStatusError):
        tools._request("GET", "https://example.test/x")


# ---------- arXiv ----------
def test_arxiv_parses_atom_and_normalises(monkeypatch):
    calls = patch_http(monkeypatch, response(text=ATOM))
    records = json.loads(tools.arxiv_search.invoke({"query": 'world "model": AND', "max_results": 99}))
    assert [r["id"] for r in records] == ["2501.00001", "hep-th/9901001"]
    assert records[0] == {"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                          "title": "A Title with breaks", "summary": "Some summary text."}
    params = calls[0][2]["params"]
    assert calls[0][1] == tools.ARXIV_URL and calls[0][1].startswith("https://")
    assert params["search_query"] == "all:world AND all:model AND all:AND"
    assert params["max_results"] == 30 and params["sortBy"] == "submittedDate"


@pytest.mark.parametrize("query", ["", "   ", '"": ::', "“”"])
def test_arxiv_empty_query_makes_no_request(monkeypatch, query):
    calls = patch_http(monkeypatch, response(text=ATOM))
    assert tools.arxiv_search.invoke({"query": query}) == "NO RESULTS"
    assert calls == []


def test_arxiv_no_entries_and_malformed_xml(monkeypatch):
    patch_http(monkeypatch, response(text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    assert tools.arxiv_search.invoke({"query": "x"}) == "NO RESULTS"
    patch_http(monkeypatch, response(text="<not xml"))
    assert tools.arxiv_search.invoke({"query": "x"}).startswith("ERROR: ParseError")


def test_arxiv_spaces_calls_three_seconds_apart(monkeypatch, no_sleep):
    patch_http(monkeypatch, response(text=ATOM))
    clock = iter([100.0, 101.0])  # one monotonic() read per call: the second call is 1s after the first
    monkeypatch.setattr(tools.time, "monotonic", lambda: next(clock))
    tools.arxiv_search.invoke({"query": "a"})
    tools.arxiv_search.invoke({"query": "b"})
    assert no_sleep == [2.0]


def test_arxiv_retries_429_then_succeeds_and_gives_up_with_error(monkeypatch, no_sleep):
    calls = patch_http(monkeypatch, response(429), response(text=ATOM))
    assert json.loads(tools.arxiv_search.invoke({"query": "x"}))
    assert len(calls) == 2
    patch_http(monkeypatch, response(429))
    out = tools.arxiv_search.invoke({"query": "x"})
    assert out.startswith("ERROR: RetryableError") and "429" in out


# ---------- Hugging Face ----------
HF_ITEMS = [
    {"paper": {"id": "2501.1", "title": "Low  vote", "summary": "about diffusion", "upvotes": 3, "githubRepo": "https://g/x",
               "githubStars": 9, "publishedAt": "2025-01-03T00:00:00.000Z"}},
    {"paper": {"id": "2501.2", "title": "High vote", "summary": "about diffusion models", "upvotes": 50,
               "publishedAt": "2025-01-04T00:00:00.000Z", "ai_summary": "short ai"}},
    {"paper": {"title": "no id"}},
    "garbage",
    {"paper": {"id": "2501.3", "title": "Other", "summary": "robots", "upvotes": 100}},
]


def test_hf_daily_filters_sorts_and_skips_invalid(monkeypatch):
    calls = patch_http(monkeypatch, response(json_body=HF_ITEMS))
    records = json.loads(tools.hf_daily_papers.invoke({"limit": 500, "date": "2025-01-04", "keyword": "DIFFUSION"}))
    assert [r["id"] for r in records] == ["2501.2", "2501.1"]
    assert records[1]["url"] == "https://huggingface.co/papers/2501.1" and records[1]["github"] == "https://g/x"
    assert records[0]["summary"] == "about diffusion models"  # the daily endpoint keeps the plain summary
    assert set(records[0]) == {"id", "url", "published", "title", "summary", "upvotes", "github", "stars"}
    assert calls[0][2]["params"] == {"limit": 100, "date": "2025-01-04"}


def test_hf_daily_no_match_and_bad_payload(monkeypatch):
    patch_http(monkeypatch, response(json_body=HF_ITEMS))
    assert tools.hf_daily_papers.invoke({"keyword": "zzz"}) == "NO RESULTS"
    patch_http(monkeypatch, response(json_body={"error": "nope"}))
    assert tools.hf_daily_papers.invoke({}).startswith("ERROR:")
    patch_http(monkeypatch, response(text="not json"))
    assert tools.hf_daily_papers.invoke({}).startswith("ERROR:")


def test_hf_search_prefers_ai_summary(monkeypatch):
    calls = patch_http(monkeypatch, response(json_body=HF_ITEMS))
    records = json.loads(tools.hf_search_papers.invoke({"query": "diffusion", "limit": 5}))
    assert {r["id"]: r["summary"] for r in records}["2501.2"] == "short ai"
    assert calls[0][1] == tools.HF_SEARCH_URL and calls[0][2]["params"] == {"q": "diffusion", "limit": 5}


def test_hf_search_empty_cases(monkeypatch):
    calls = patch_http(monkeypatch, response(json_body=[]))
    assert tools.hf_search_papers.invoke({"query": " "}) == "NO RESULTS" and calls == []
    assert tools.hf_search_papers.invoke({"query": "x"}) == "NO RESULTS"
    patch_http(monkeypatch, response(500))
    assert tools.hf_search_papers.invoke({"query": "x"}).startswith("ERROR: RetryableError")


# ---------- Exa ----------
def sse(result=None, error=None):
    body = {"jsonrpc": "2.0", "id": 1}
    body.update({"result": result} if error is None else {"error": error})
    return response(text="event: message\ndata: " + json.dumps(body) + "\n\n")


def text_result(text, **extra):
    return {"content": [{"type": "text", "text": text}], **extra}


def test_exa_search_parses_sse_and_sends_contract(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    calls = patch_http(monkeypatch, sse(text_result("Title: T\nURL: https://x.org")))
    assert tools.web_search.invoke({"query": "world models", "num_results": 3}) == "Title: T\nURL: https://x.org"
    method, url, kw = calls[0]
    assert (method, url) == ("POST", tools.EXA_URL)
    assert kw["headers"]["Accept"] == "application/json, text/event-stream"
    assert kw["json"]["method"] == "tools/call"
    assert kw["json"]["params"] == {"name": "web_search_exa", "arguments": {
        "query": "world models", "objective": "Find pages about: world models", "numResults": 3}}


def test_exa_key_goes_in_url_and_is_redacted_from_errors(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "sekret-key/123")
    calls = patch_http(monkeypatch, httpx.ConnectError("failed for https://mcp.exa.ai/mcp?exaApiKey=sekret-key%2F123"))
    out = tools.web_search.invoke({"query": "x"})
    assert "exaApiKey=sekret-key%2F123" in calls[0][1]
    assert out.startswith("ERROR:") and "sekret" not in out and "%2F123" not in out


def test_exa_key_redacted_on_http_error(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "abcdef123456")
    req = httpx.Request("POST", "https://mcp.exa.ai/mcp?exaApiKey=abcdef123456")
    patch_http(monkeypatch, httpx.Response(401, request=req))
    out = tools.web_fetch.invoke({"url": "https://a.org"})
    assert out.startswith("ERROR: HTTPStatusError") and "abcdef123456" not in out


def test_exa_rate_limit_via_meta_flag_is_retried_not_returned(monkeypatch, no_sleep):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    limited = sse(text_result("Slow down", _meta={"ai.exa/rateLimited": True}))
    calls = patch_http(monkeypatch, limited, sse(text_result("real content")))
    assert tools.web_search.invoke({"query": "x"}) == "real content"
    assert len(calls) == 2 and len(no_sleep) == 1


def test_exa_rate_limit_via_text_and_persistent_failure(monkeypatch, no_sleep):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    patch_http(monkeypatch, sse(text_result("You have hit the rate limit. Please retry in 20 seconds.")))
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR: RetryableError") and "rate limit" in out.lower()
    assert len(no_sleep) == tools.EXA_RETRY["attempts"] - 1


def test_exa_long_page_mentioning_rate_limits_is_not_misdetected(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    page = "An article about API rate limit design. " * 100
    patch_http(monkeypatch, sse(text_result(page)))
    assert tools.web_search.invoke({"query": "x"}) == page.strip()


def test_exa_jsonrpc_error_and_empty_and_garbage(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    patch_http(monkeypatch, sse(error={"code": -32602, "message": "Invalid params"}))
    assert tools.web_search.invoke({"query": "x"}) == "ERROR: ValueError: Exa JSON-RPC error: Invalid params"
    patch_http(monkeypatch, sse(text_result("   ")))
    assert tools.web_search.invoke({"query": "x"}) == "NO RESULTS"
    patch_http(monkeypatch, response(text="event: message\ndata: not-json\n\n"))
    assert tools.web_search.invoke({"query": "x"}).startswith("ERROR:")
    assert tools.web_search.invoke({"query": "  "}) == "NO RESULTS"


def test_exa_jsonrpc_rate_limit_error_is_retried(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    patch_http(monkeypatch, sse(error={"message": "Rate limit exceeded"}), sse(text_result("fine")))
    assert tools.web_search.invoke({"query": "x"}) == "fine"


def test_exa_accepts_plain_json_body(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    patch_http(monkeypatch, response(json_body={"jsonrpc": "2.0", "id": 1, "result": text_result("plain")}))
    assert tools.web_search.invoke({"query": "x"}) == "plain"


def test_web_fetch_sends_urls_array_and_truncates(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    calls = patch_http(monkeypatch, sse(text_result("a" * 20000)))
    out = tools.web_fetch.invoke({"url": " https://arxiv.org/abs/1803.10122 "})
    assert len(out) == tools.FETCH_CHARS
    assert calls[0][2]["json"]["params"] == {"name": "web_fetch_exa", "arguments": {"urls": ["https://arxiv.org/abs/1803.10122"]}}
    assert tools.web_fetch.invoke({"url": ""}) == "NO RESULTS"


def test_no_tool_ever_raises(monkeypatch):
    patch_http(monkeypatch, RuntimeError("kaboom"))
    for t, args in [(tools.arxiv_search, {"query": "x"}), (tools.hf_daily_papers, {}),
                    (tools.hf_search_papers, {"query": "x"}), (tools.web_search, {"query": "x"}),
                    (tools.web_fetch, {"url": "https://a.org"})]:
        assert t.invoke(args).startswith("ERROR: RuntimeError")
