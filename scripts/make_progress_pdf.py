import re
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Arial-Bold'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def parse_markdown(path: Path, styles):
    story = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), styles["Title"]))
            story.append(Spacer(1, 0.25 * cm))
        elif line.startswith("## "):
            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph(clean_inline(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(lines[index][2:])
                index += 1
            for item in items:
                story.append(Paragraph("• " + clean_inline(item), styles["Body"]))
            continue
        elif line.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].startswith("| "):
                table_lines.append(lines[index])
                index += 1
            rows = []
            for table_line in table_lines:
                cells = [clean_inline(cell.strip()) for cell in table_line.strip("|").split("|")]
                if set("".join(cells)) <= {"-", ":", " "}:
                    continue
                rows.append([Paragraph(cell, styles["TableCell"]) for cell in cells])
            if rows:
                table = Table(rows, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.2 * cm))
            continue
        else:
            paragraph = [line]
            while index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].startswith(("#", "- ", "| ")):
                index += 1
                paragraph.append(lines[index].strip())
            story.append(Paragraph(clean_inline(" ".join(paragraph)), styles["Body"]))
        index += 1
    return story


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/progress_report.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/progress_report.pdf"))
    args = parser.parse_args()

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(font_path)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold_path)))
        font_name = "Arial"
        bold_name = "Arial-Bold"
    else:
        font_name = "Helvetica"
        bold_name = "Helvetica-Bold"

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName=bold_name, fontSize=18, leading=22),
        "Heading2": ParagraphStyle("Heading2", parent=base["Heading2"], fontName=bold_name, fontSize=13, leading=16, spaceAfter=6),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName=font_name, fontSize=9.5, leading=13, spaceAfter=5),
        "TableCell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName=font_name, fontSize=8, leading=10),
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm)
    story = parse_markdown(args.input, styles)
    doc.build(story)
    print(out)


if __name__ == "__main__":
    main()
