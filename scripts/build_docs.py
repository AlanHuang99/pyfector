"""Build the API documentation.

Inputs: package docstrings and docs/templates.
Output: pdoc HTML in docs/ or the supplied output directory.
Notes: NumPy sections use headings scoped to their API member.
"""

from pathlib import Path
import argparse
import re

import pdoc


ROOT = Path(__file__).resolve().parents[1]
MODULES = ["pyfector"] + [
    f"pyfector.{name}" for name in
    ("diagnostics", "inference", "panel", "estimators", "cv", "plotting", "backend", "linalg")
]


def scope_headings(html, namespace):
    """Give each member's docstring headings unique fragment identifiers."""
    if not namespace:
        return html
    ids = re.findall(r'<h[1-6]\b[^>]*\bid="([^"]+)"', html)
    for identifier in ids:
        scoped = f"{namespace}--{identifier}"
        html = html.replace(f'id="{identifier}"', f'id="{scoped}"')
        html = html.replace(f'href="#{identifier}"', f'href="#{scoped}"')
    return html


def entrypoint_links(html, modulename):
    """Link the public function alias without confusing it with its module."""
    target = "#fect" if modulename == "pyfector" else "../pyfector.html#fect"
    return re.sub(
        r'<a href="[^"]*"([^>]*)>pyfector\.fect</a>',
        lambda match: f'<a href="{target}"{match[1]}>pyfector.fect</a>',
        str(html),
    )


def build(output):
    pdoc.render.configure(docformat="numpy", template_directory=ROOT / "docs" / "templates")
    pdoc.render.env.filters["scope_headings"] = scope_headings
    pdoc.render.env.filters["entrypoint_links"] = entrypoint_links
    pdoc.pdoc(*MODULES, output_directory=output)
    for path in output.rglob("*.html"):
        text = path.read_text()
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    build(parser.parse_args().output_dir)
