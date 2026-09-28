"""Generate pip find-links pages from the wheels of all published releases.

Usage: python make_wheel_index.py OWNER/REPO OUTPUT_DIR

Writes one page per torch minor x CUDA major (e.g. ``torch-2.12+cu13.html``,
derived from the wheel's local version tag ``+pt212cu130``) and an
``index.html`` listing them. Wheels are built per CUDA major (CUDA minor-version
compatibility), so the page serves every torch build of that major (cu130,
cu132, ...).
"""

import html
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

LOCAL_TAG = re.compile(r"\+pt(\d)(\d+)cu(\d+)-")


def _get(url: str) -> list[dict]:
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _releases(repo: str) -> list[dict]:
    releases: list[dict] = []
    page = 1
    while batch := _get(
        f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
    ):
        releases += batch
        page += 1
    return [r for r in releases if not r["draft"]]


def _page(title: str, links: list[tuple[str, str]]) -> str:
    body = "\n".join(
        f'<a href="{html.escape(href)}">{html.escape(text)}</a><br>'
        for text, href in links
    )
    return (
        '<!DOCTYPE html>\n<html>\n<head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head>\n"
        f"<body>\n<h1>{html.escape(title)}</h1>\n{body}\n</body>\n</html>\n"
    )


def main() -> None:
    """Write the index pages."""
    repo, out_dir = sys.argv[1], Path(sys.argv[2])
    pages: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for release in _releases(repo):
        for asset in release["assets"]:
            name = asset["name"]
            match = LOCAL_TAG.search(name)
            if not name.endswith(".whl") or match is None:
                continue
            major, minor, cuda = match.groups()
            cuda_major = cuda[:-1]  # "130" -> "13"
            href = asset["browser_download_url"]
            digest = asset.get("digest") or ""
            if digest.startswith("sha256:"):
                href += "#sha256=" + digest.removeprefix("sha256:")
            pages[f"torch-{major}.{minor}+cu{cuda_major}"].append((name, href))

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.html"):
        old.unlink()
    for combo, links in pages.items():
        (out_dir / f"{combo}.html").write_text(
            _page(f"phipict wheels for {combo}", sorted(links))
        )
    index = [(combo, f"{combo}.html") for combo in sorted(pages)]
    (out_dir / "index.html").write_text(_page("phipict wheels", index))
    print(f"Wrote {len(pages)} index page(s) to {out_dir}")


if __name__ == "__main__":
    main()
