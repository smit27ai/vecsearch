"""Write plain text to a simple PDF, with no PDF library.

Used to turn rag/sample/dsa_notes.md into a real PDF so the pipeline's PDF
path is exercised end to end, and by the tests to make PDFs on the fly.
Supports ASCII text only (Helvetica, one font size).

Usage:
    python -m rag.make_pdf rag/sample/dsa_notes.md rag/sample/dsa_notes.pdf
"""

import sys
import textwrap

LINES_PER_PAGE = 54
CHARS_PER_LINE = 95


def _escape(line):
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_to_pages(text):
    """Wrap text into lines and split the lines into pages.

    A markdown heading ("# ...") starts a new page when the current page
    is more than two-thirds full, so sections rarely split badly.
    """
    lines = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            if len(lines) % LINES_PER_PAGE > LINES_PER_PAGE * 2 // 3:
                lines += [""] * (LINES_PER_PAGE - len(lines) % LINES_PER_PAGE)
            lines.append(raw.lstrip("#").strip().upper())
        elif raw.strip():
            # break_on_hyphens=False: a line ending in "in-" would be read back
            # as a word split across lines and joined into "inorder"
            lines += textwrap.wrap(raw, CHARS_PER_LINE, break_on_hyphens=False) or [""]
        else:
            lines.append("")
    return [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


def write_pdf(pages, path):
    """pages: list of pages, each a list of text lines."""
    objects = [None, None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_ids = []
    for lines in pages:
        body = "BT /F1 10 Tf 13 TL 50 800 Td\n"
        body += "".join(f"({_escape(line)}) Tj T*\n" for line in lines) + "ET"
        stream = body.encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_id)
        page_ids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="ascii") as f:
        pages = text_to_pages(f.read())
    write_pdf(pages, dst)
    print(f"wrote {dst}: {len(pages)} pages")
