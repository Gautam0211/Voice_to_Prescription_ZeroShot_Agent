"""
Converts the same HTML prescription sheet shown on-screen into a PDF for
download. Uses xhtml2pdf instead of weasyprint deliberately — weasyprint
needs system-level Cairo/Pango libraries that are painful to install on
Windows, while xhtml2pdf is pure-Python and installs cleanly everywhere,
including Streamlit Cloud.

xhtml2pdf only supports a subset of CSS (no flexbox, limited modern
properties), so this builds a simplified, PDF-safe version of the sheet
rather than reusing the on-screen HTML directly — flexbox layouts in the
on-screen version would silently collapse in the PDF output.
"""

from io import BytesIO

from xhtml2pdf import pisa


def build_pdf_html(patient_name: str, age: str, gender: str, doctor_name: str, date: str, medicines: list[dict]) -> str:
    rows_html = ""
    for m in medicines:
        strength = f" &middot; {m['strength']}" if m.get("strength") else ""
        freq = f" &middot; {m['frequency']}" if m.get("frequency") else ""
        dur = f" &middot; {m['duration']}" if m.get("duration") else ""
        instr = f"<br/><span style='color:#6B7280;font-size:9pt;'>{m['instructions']}</span>" if m.get("instructions") else ""
        rows_html += (
            f'<div style="padding:8px 0;border-bottom:1px solid #DAD4C6;">'
            f'<span style="font-weight:bold;">{m["drug_name"]}</span>{strength}{freq}{dur}'
            f"{instr}</div>"
        )

    return f"""
    <html>
    <head>
    <style>
        body {{ font-family: Helvetica, sans-serif; color: #1B2A4A; padding: 24px; }}
        .header {{ border-bottom: 2px solid #1B2A4A; padding-bottom: 10px; margin-bottom: 14px; }}
        .mark {{ font-size: 26pt; color: #2F6F62; font-weight: bold; }}
        .title {{ font-size: 18pt; font-weight: bold; }}
        .meta {{ font-size: 10pt; color: #6B7280; margin-bottom: 16px; }}
        .meta td {{ padding-right: 24px; }}
    </style>
    </head>
    <body>
        <div class="header">
            <span class="mark">Rx</span>
            <span class="title">Prescription</span>
        </div>
        <table class="meta">
            <tr>
                <td>Patient: {patient_name or '-'}</td>
                <td>Age: {age or '-'}</td>
                <td>Gender: {gender or '-'}</td>
            </tr>
            <tr>
                <td>Doctor: {doctor_name or '-'}</td>
                <td>Date: {date or '-'}</td>
            </tr>
        </table>
        {rows_html}
    </body>
    </html>
    """


def html_to_pdf_bytes(html: str) -> bytes:
    buffer = BytesIO()
    pisa.CreatePDF(src=html, dest=buffer)
    return buffer.getvalue()