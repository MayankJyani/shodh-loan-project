import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import textwrap

def py_to_pdf(src_path, out_path=None, font_name="Courier", font_size=9):
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src)
    if out_path is None:
        out_path = src.with_suffix('.pdf')
    out = Path(out_path)

    lines = src.read_text(encoding='utf-8', errors='replace').splitlines()
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    left_margin = 15 * mm
    right_margin = 15 * mm
    top_margin = 15 * mm
    bottom_margin = 15 * mm
    usable_width = width - left_margin - right_margin
    x = left_margin
    y = height - top_margin

    c.setFont(font_name, font_size)
    # approximate max chars per line (monospace)
    avg_char_width = c.stringWidth("M", font_name, font_size)
    max_chars = max(40, int(usable_width / avg_char_width))

    leading = font_size * 1.2

    for orig in lines:
        wrapped = textwrap.wrap(orig, width=max_chars, replace_whitespace=False) or ['']
        for w in wrapped:
            if y - leading < bottom_margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - top_margin
            c.drawString(x, y, w)
            y -= leading

    c.save()
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python py_to_pdf.py <input.py> [output.pdf]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    pdf_path = py_to_pdf(inp, outp)
    print("Saved PDF to", pdf_path)