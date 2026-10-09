"""Offline tests for check_citations.check (and agreement with the provided finalize_citations.py)."""
import pytest

from check_citations import check
from finalize_citations import finalize

SOURCES = [
    {"n": 1, "id": "a", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01", "source": "arxiv"},
    {"n": 2, "id": "b", "url": "https://huggingface.co/papers/2501.00002", "title": "B", "date": "2025-01-02", "source": "hf-search"},
    {"n": 3, "id": "c", "url": "https://example.org/c", "title": "C", "date": "2025-01-03", "source": "web"},
]
REFS = ("## References\n[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)\n"
        "[2] B. hf-search. https://huggingface.co/papers/2501.00002 (2025-01-02)\n"
        "[3] C. web. https://example.org/c (2025-01-03)\n")


def report(body="Claim one [1]. Claim two [2][3].\n", refs=REFS):
    return f"# T\n\n## TL;DR\n{body}\n{refs}"


def has(problems, text):
    return any(text in p for p in problems)


def test_valid_report_passes():
    assert check(report(), SOURCES) == []


def test_empty_or_invalid_sources():
    assert check(report(), []) == ["no sources in sources.json"]
    assert check(report(), None) == ["no sources in sources.json"]


def test_bad_source_entries():
    bad = [{"n": "1", "url": "https://a.org"}, {"n": 2, "url": "ftp://x"}, {"n": 3, "url": "https://a.org"},
           {"n": 4, "url": "https://a.org"}, {"n": True, "url": "https://t.org"}, "oops", {"n": 3, "url": "https://z.org"}]
    problems = check(report(), bad)
    assert has(problems, "not an integer") and has(problems, "invalid url") and has(problems, "duplicates the url")
    assert has(problems, "not an object") and has(problems, "appears twice")


def test_missing_references_heading():
    assert has(check("# T\nClaim [1][2][3].\n", SOURCES), "missing the '## References' heading")


def test_citation_to_unknown_source_and_uncited_source():
    problems = check(report("Only [1] and [9]."), SOURCES)
    assert has(problems, "[9] cited but missing") and has(problems, "source [2] never cited") and has(problems, "source [3] never cited")


def test_numbers_in_references_do_not_count_as_citations():
    assert has(check(report("Only [1] and [2]."), SOURCES), "source [3] never cited")


@pytest.mark.parametrize("body", ["A [1, 2, 3].", "A [1-3].", "A [1,2] and [3]", "A [1–3]", "A [1][2], [3]"])
def test_grouped_citations_are_expanded(body):
    assert check(report(body), SOURCES) == []


def test_code_and_markdown_links_are_not_citations():
    body = "Real [1][2][3]. Inline `[7]` and\n```\n[8]\n```\nand a link [9](https://x.org) and [1](https://y.org).\n"
    assert check(report(body), SOURCES) == []
    assert has(check(report("Only [1][2] and `[3]` and [3](https://x.org)."), SOURCES), "source [3] never cited")


def test_missing_and_duplicate_reference_lines():
    problems = check(report(refs="## References\n[1] A. https://arxiv.org/abs/2501.00001\n"
                                 "[1] A again. https://arxiv.org/abs/2501.00001\n[2] B. https://huggingface.co/papers/2501.00002\n"), SOURCES)
    assert has(problems, "source [3] has no line") and has(problems, "2 lines numbered [1]")


def test_reference_line_for_unknown_number():
    refs = REFS + "[4] D. https://example.org/d\n"
    assert has(check(report(refs=refs), SOURCES), "line [4] that is not in sources.json")


def test_bundled_reference_line_is_rejected():
    refs = REFS.replace("[3] C. web. https://example.org/c (2025-01-03)",
                        "[3] C. https://example.org/c; D. https://example.org/d")
    assert has(check(report(refs=refs), SOURCES), "exactly one URL, found 2")


def test_reference_line_without_or_with_wrong_url():
    refs = REFS.replace("https://example.org/c", "")
    assert has(check(report(refs=refs), SOURCES), "exactly one URL, found 0")
    refs = REFS.replace("https://example.org/c", "https://example.org/other")
    assert has(check(report(refs=refs), SOURCES), "differs from the sources.json url")


def test_trailing_punctuation_after_url_is_tolerated():
    assert check(report(refs=REFS.replace("(2025-01-03)", "(2025-01-03).").replace("c (", "c. (")), SOURCES) == []


def test_continuation_line_with_url_is_flagged():
    assert has(check(report(refs=REFS + "see also https://x.org\n"), SOURCES), "does not start with [n]")


def test_finalizer_output_always_validates():
    body = "Intro [3, 1]. More [2-3]. Code `[5]`.\n"
    sources = SOURCES + [{"n": 4, "id": "d", "url": "https://example.org/c", "title": "dup url", "source": "web"}]
    new_report, new_sources, problems = finalize(f"# T\n{body}", sources)
    assert problems == [] and check(new_report, new_sources) == []
