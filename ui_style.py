"""
Visual styling for the clinical UI — kept separate from app.py so the
design system is one file to tune, not scattered inline strings.

Design intent: a prescription pad / clinical letterhead feel, not a
generic SaaS dashboard. One accent color, one serif mark, status shown
as a left-edge stripe on each row rather than badges everywhere.
"""

import textwrap

import streamlit as st

INK = "#1B2A4A"       # deep navy — headings, primary text
PAPER = "#FBF9F5"     # warm off-white — page background
LINE = "#DAD4C6"      # hairline rule / border color
ACCENT = "#2F6F62"    # clinical teal — the one accent color, used sparingly
FLAG = "#B65C1F"      # amber-rust — flagged rows only
FLAG_BG = "#FBF0E6"
MUTED = "#6B7280"      # secondary text


def inject_base_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        .stApp {{
            background-color: {PAPER};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {INK};
        }}

        /* Letterhead */
        .rx-letterhead {{
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            border-bottom: 2px solid {INK};
            padding-bottom: 0.75rem;
            margin-bottom: 0.25rem;
        }}
        .rx-mark {{
            font-family: 'Source Serif 4', serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: {ACCENT};
            line-height: 1;
        }}
        .rx-title {{
            font-family: 'Source Serif 4', serif;
            font-size: 1.7rem;
            font-weight: 600;
            color: {INK};
        }}
        .rx-subtitle {{
            font-size: 0.85rem;
            color: {MUTED};
            margin-top: -0.3rem;
            margin-bottom: 1.4rem;
        }}

        /* Section labels — small, not full caps-every-label overuse */
        .rx-section-label {{
            font-size: 0.78rem;
            color: {MUTED};
            letter-spacing: 0.02em;
            margin-bottom: 0.4rem;
        }}

        /* Review row wrapper: colored left stripe communicates status */
        .rx-row-ok {{
            border-left: 3px solid {ACCENT};
            padding-left: 0.9rem;
            margin-bottom: 0.4rem;
        }}
        .rx-row-flag {{
            border-left: 3px solid {FLAG};
            background: {FLAG_BG};
            padding: 0.6rem 0.9rem 0.2rem 0.9rem;
            border-radius: 2px;
            margin-bottom: 0.4rem;
        }}
        .rx-flag-note {{
            font-size: 0.82rem;
            color: {FLAG};
            margin: 0.2rem 0 0.5rem 0;
        }}
        .rx-match-note {{
            font-size: 0.78rem;
            color: {MUTED};
            margin-top: -0.6rem;
            margin-bottom: 0.6rem;
        }}

        /* Buttons: squared-off, letterhead-appropriate, not bubbly SaaS */
        .stButton > button {{
            border-radius: 3px;
            border: 1px solid {INK};
            color: {INK};
            font-weight: 500;
        }}
        .stButton > button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
        }}
        .stButton > button[kind="primary"] {{
            background-color: {ACCENT};
            border-color: {ACCENT};
            color: {PAPER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _html(text: str) -> str:
    """
    Strips ALL leading whitespace from every line. Streamlit's Markdown
    parser treats any line starting with whitespace as a fenced code
    block rather than rendering it as HTML — a plain textwrap.dedent()
    isn't enough for nested HTML, since inner tags keep relative
    indentation. HTML doesn't care about whitespace between tags, so
    flattening every line is safe here.
    """
    return "\n".join(line.lstrip() for line in text.strip().splitlines())


def letterhead(title: str, subtitle: str) -> None:
    st.markdown(
        _html(
            f"""
            <div class="rx-letterhead">
                <span class="rx-mark">℞</span>
                <span class="rx-title">{title}</span>
            </div>
            <div class="rx-subtitle">{subtitle}</div>
            """
        ),
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="rx-section-label">{text}</div>', unsafe_allow_html=True)


def render_prescription_sheet(patient_name: str, doctor_name: str, date: str, medicines: list[dict]) -> str:
    """Builds a static HTML prescription sheet for the finalized output."""
    rows = ""
    for m in medicines:
        strength = f" &middot; {m['strength']}" if m.get("strength") else ""
        freq = f" &middot; {m['frequency']}" if m.get("frequency") else ""
        dur = f" &middot; {m['duration']}" if m.get("duration") else ""
        instr = f"<br><span style='color:{MUTED};font-size:0.82rem'>{m['instructions']}</span>" if m.get("instructions") else ""
        rows += (
            f'<div style="padding:0.7rem 0;border-bottom:1px solid {LINE};">'
            f'<span style="font-weight:600;">{m["drug_name"]}</span>{strength}{freq}{dur}'
            f'{instr}</div>'
        )

    return _html(
        f"""
        <div style="border:1px solid {LINE}; padding:1.6rem; background:white; border-radius:2px;">
            <div class="rx-letterhead" style="margin-bottom:1rem;">
                <span class="rx-mark">℞</span>
                <span class="rx-title" style="font-size:1.3rem;">Prescription</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:{MUTED}; margin-bottom:1rem;">
                <span>Patient: {patient_name or '—'}</span>
                <span>Doctor: {doctor_name or '—'}</span>
                <span>Date: {date or '—'}</span>
            </div>
            {rows}
        </div>
        """
    )