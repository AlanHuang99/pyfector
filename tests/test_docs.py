"""Check rendered documentation navigation and NumPy parameter sections."""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import runpy
from urllib.parse import unquote, urlsplit

import pytest


pytest.importorskip("pdoc")


class _Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.links = []
        self.anchor_text = []
        self.current_link = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
            self.current_link = [attrs["href"], ""]

    def handle_data(self, text):
        if self.current_link is not None:
            self.current_link[1] += text

    def handle_endtag(self, tag):
        if tag == "a" and self.current_link is not None:
            self.anchor_text.append(tuple(self.current_link))
            self.current_link = None


def test_documentation_renders_and_local_links_resolve(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_docs.py"
    runpy.run_path(str(script))["build"](tmp_path)
    pages = {path.resolve(): _Page(path.read_text()) for path in tmp_path.rglob("*.html")}
    for name in ("fect", "diagnostics", "inference", "panel", "estimators", "cv"):
        assert (tmp_path / "pyfector" / f"{name}.html").resolve() in pages
    for path, page in pages.items():
        assert not [identifier for identifier, count in Counter(page.ids).items() if count > 1], path
        for href in page.links:
            assert href, path
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            assert target in pages, (path, href)
            if parts.fragment:
                assert unquote(parts.fragment) in pages[target].ids, (path, href)
    html = (tmp_path / "pyfector.html").read_text()
    assert 'id="fect--parameters"' in html
    assert "Usage::" not in html
    assert '<p>::</p>' not in (tmp_path / "pyfector" / "fect.html").read_text()
    assert 'href="https://github.com/xuyiqing/fect"' in html
    assert '<a href="pyfector/diagnostics.html">diagnostics</a>' in html
    assert '<a href="pyfector/inference.html">inference</a>' in html
    for name in ("pyfector.html", "pyfector/fect.html", "pyfector/plotting.html"):
        page = pages[(tmp_path / name).resolve()]
        references = [(href, label) for href, label in page.anchor_text if label == "pyfector.fect"]
        assert references, name
        assert all(href.endswith("#fect") for href, _ in references), (name, references)
    for name in ("inference", "panel", "cv_result"):
        assert f'<section id="FectResult.{name}"' not in html  # dataclass members use divs
        assert f'id="FectResult.{name}"' in html
    assert "Bootstrap or jackknife inference results" in html
