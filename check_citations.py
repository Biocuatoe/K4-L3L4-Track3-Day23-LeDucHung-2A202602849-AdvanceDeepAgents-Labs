"""check_citations.py - validates the citations of a report.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It exits 0 and prints "OK: ..." when the report is consistent, else prints each problem and exits 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)                      # odd split segments are code
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")           # [3]  [1, 2]  [1-3]; not the link [3](url)
_SPAN = re.compile(r"(\d+)\s*[–-]\s*(\d+)")
_REF_LINE = re.compile(r"^\[(\d+)\]")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_MAX_SPAN = 200


def _expand(group):
    """'1, 3-5' -> [1, 3, 4, 5]"""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = _SPAN.fullmatch(part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= _MAX_SPAN else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited_numbers(body):
    cited = set()
    for i, segment in enumerate(_CODE.split(body)):
        if i % 2:  # code spans and fenced blocks never count
            continue
        for match in _CITATION.finditer(segment):
            cited.update(_expand(match.group(1)))
    return cited


def _check_sources(sources):
    """Return (problems, {n: url}) for the sources.json entries."""
    problems, by_n, seen_urls = [], {}, {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n, url = entry.get("n"), entry.get("url")
        valid_n = isinstance(n, int) and not isinstance(n, bool)
        if not valid_n:
            problems.append(f"source n={n!r} is not an integer")
        elif n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}] has an invalid url: {url!r}")
        elif url in seen_urls:
            problems.append(f"source [{n}] duplicates the url of source [{seen_urls[url]}]: {url}")
        else:
            seen_urls[url] = n
        if valid_n:
            by_n.setdefault(n, url)
    return problems, by_n


def _check_references(references, by_n):
    problems, lines_by_n = [], {}
    for line in (ln.strip() for ln in references.splitlines()):
        match = _REF_LINE.match(line)
        if match:
            lines_by_n.setdefault(int(match.group(1)), []).append(line)
        elif _URL.search(line):
            problems.append(f"reference line does not start with [n]: {line[:80]}")
    problems += [f"source [{n}] has no line in References" for n in sorted(by_n.keys() - lines_by_n.keys())]
    problems += [f"References has a line [{n}] that is not in sources.json" for n in sorted(lines_by_n.keys() - by_n.keys())]
    for n, lines in sorted(lines_by_n.items()):
        if len(lines) > 1:
            problems.append(f"References has {len(lines)} lines numbered [{n}]")
        if n not in by_n:
            continue
        for line in lines:
            urls = [u.rstrip(".,;:") for u in _URL.findall(line)]
            if len(urls) != 1:
                problems.append(f"reference [{n}] must contain exactly one URL, found {len(urls)}")
            elif urls[0] != by_n[n]:
                problems.append(f"reference [{n}] url {urls[0]} differs from the sources.json url {by_n[n]}")
    return problems


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK). Grouped citations [1, 2] and [1-3] are expanded."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems, by_n = _check_sources(sources)
    headings = list(_REF_HEADING.finditer(report_text))
    if headings:
        body, references = report_text[:headings[-1].start()], report_text[headings[-1].end():]
    else:
        problems.append("missing the '## References' heading")
        body, references = report_text, ""
    cited = _cited_numbers(body)
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys())]
    problems += [f"source [{n}] never cited in the report body" for n in sorted(by_n.keys() - cited)]
    if headings:
        problems += _check_references(references, by_n)
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
