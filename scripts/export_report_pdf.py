"""Convert Markdown report to PDF using ReportLab."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = PROJECT_ROOT / "docs" / "report.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "report.pdf"


def format_inline(text: str) -> str:
    """Convert a subset of Markdown inline styles to ReportLab-friendly markup."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"`([^`]+)`", r"<font name='Courier'>\1</font>", text
    )
    return text


def build_styles():
    styles = getSampleStyleSheet()
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 22
    styles["Heading1"].spaceAfter = 8

    styles["Heading2"].fontSize = 14
    styles["Heading2"].leading = 18
    styles["Heading2"].spaceAfter = 6

    styles["Heading3"].fontSize = 12
    styles["Heading3"].leading = 14
    styles["Heading3"].spaceAfter = 4

    styles["BodyText"].fontSize = 10.5
    styles["BodyText"].leading = 14

    styles.add(
        ParagraphStyle(
            name="CustomBullet",
            parent=styles["BodyText"],
            leftIndent=16,
            bulletIndent=8,
            bulletFontSize=8,
        )
    )
    return styles


def markdown_to_story(markdown: str):
    styles = build_styles()
    heading_styles = {
        1: styles["Heading1"],
        2: styles["Heading2"],
        3: styles["Heading3"],
    }
    story = []
    list_items = []

    def flush_list():
        nonlocal list_items
        if list_items:
            story.append(
                ListFlowable(
                    list_items,
                    bulletType="bullet",
                    leftIndent=12,
                    bulletFontName="Helvetica",
                    bulletFontSize=8,
                )
            )
            list_items = []
            story.append(Spacer(1, 6))

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line:
            flush_list()
            story.append(Spacer(1, 6))
            continue
        if line.strip() == "---":
            flush_list()
            story.append(Spacer(1, 12))
            continue
        heading_match = re.match(r"^(#+)\s+(.*)", line)
        if heading_match:
            flush_list()
            level = len(heading_match.group(1))
            text = format_inline(heading_match.group(2).strip())
            style = heading_styles.get(level, styles["Heading3"])
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 4))
            continue
        if line.strip().startswith("- "):
            text = format_inline(line.strip()[2:].strip())
            list_items.append(ListItem(Paragraph(text, styles["CustomBullet"])))
            continue
        flush_list()
        story.append(Paragraph(format_inline(line), styles["BodyText"]))

    flush_list()
    return story


def main() -> None:
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(f"Markdown report not found at {MARKDOWN_PATH}")
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    story = markdown_to_story(markdown)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48,
    )
    doc.build(story)
    print(f"Saved PDF report to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
