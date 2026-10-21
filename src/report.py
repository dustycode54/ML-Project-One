"""Render the final report from Markdown to PDF (Ticket 07).

Stages:
  1. Convert reports/final_report.md to HTML (python-markdown, with tables and fenced code). An image
     whose alt text starts with "Figure N." becomes a <figure> with that text as its caption.
  2. Wrap it in a print stylesheet (letter paper, page numbers, no page breaks inside figures or tables).
  3. Print the HTML to reports/final_report.pdf with headless Chrome / Chromium. Set CHROME to the
     browser binary if it is not in a standard location.

The figures are referenced as reports/figures/*.svg, so run python -m src.plots first.

Run: python -m src.report
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

from .preprocess import REPORTS_DIR
from .scrapers.common import setup_logging

log = logging.getLogger("report")

SOURCE = REPORTS_DIR / "final_report.md"
OUTPUT = REPORTS_DIR / "final_report.pdf"
TITLE = "Predicting Used-Car Value Retention"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge",
]

CSS = """
@page { size: letter; margin: 0.8in 0.85in 0.8in 0.85in;
        @bottom-center { content: counter(page); font: 8.5pt "Helvetica Neue", Helvetica, Arial, sans-serif;
                         color: #898781; } }
:root { --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781; --rule: #e1e0d9; --wash: #f6f6f3; }
html { background: #ffffff; }
body { font: 10pt/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; color: var(--ink); margin: 0; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 4pt; }
h1 + p { color: var(--ink-2); margin-top: 0; }
h2 { font-size: 13.5pt; margin: 22pt 0 6pt; padding-top: 6pt; border-top: 1px solid var(--rule);
     break-after: avoid; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
code { font: 8.5pt Menlo, Consolas, monospace; background: var(--wash); padding: 0 2pt; border-radius: 2px; }
pre { background: var(--wash); padding: 8pt 10pt; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; break-inside: avoid; }
pre code { padding: 0; font-size: 8pt; }
blockquote { margin: 10pt 0; padding: 6pt 12pt; background: var(--wash); border-left: 3px solid #2a78d6; }
blockquote p { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 8.5pt;
        font-variant-numeric: tabular-nums; }
tr { break-inside: avoid; }
p:has(+ table), p:has(+ ul), p:has(+ ol), p:has(+ pre) { break-after: avoid; }
th { text-align: left; font-weight: 600; color: var(--ink-2); border-bottom: 1px solid var(--ink-2); }
th, td { padding: 3pt 6pt 3pt 0; vertical-align: top; }
td { border-bottom: 1px solid var(--rule); }
figure { margin: 12pt 0 14pt; break-inside: avoid; }
figure img { display: block; width: 100%; max-height: 3.5in; object-fit: contain; }
figcaption { font-size: 8.5pt; line-height: 1.4; color: var(--ink-2); margin-top: 4pt; }
figcaption b { color: var(--ink); }
.equation { text-align: center; font-style: italic; margin: 8pt 0; }
"""

FIGURE = re.compile(r'<p>\s*<img alt="(Figure \d+\.)([^"]*)" src="([^"]+)"\s*/?>\s*</p>')


def to_html(text):
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = FIGURE.sub(r'<figure><img src="\3" alt="\1\2"><figcaption><b>\1</b>\2</figcaption></figure>', body)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def find_chrome():
    for candidate in [os.environ.get("CHROME")] + CHROME_CANDIDATES:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return shutil.which(candidate) or candidate
    raise FileNotFoundError("Chrome / Chromium not found; set CHROME to the browser executable")


def print_pdf(html_path, pdf_path):
    subprocess.run([find_chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=120)


def run():
    html = to_html(SOURCE.read_text())
    missing = [src for src in re.findall(r'<img src="([^"]+)"', html) if not (REPORTS_DIR / src).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figures (run python -m src.plots): {missing}")
    # The HTML is written next to the Markdown so the relative figure paths resolve.
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=REPORTS_DIR, delete=False) as f:
        f.write(html)
    try:
        print_pdf(Path(f.name), OUTPUT)
    finally:
        os.unlink(f.name)
    log.info("Wrote %s (%.1f MB)", OUTPUT, OUTPUT.stat().st_size / 1e6)


if __name__ == "__main__":
    setup_logging()
    run()
