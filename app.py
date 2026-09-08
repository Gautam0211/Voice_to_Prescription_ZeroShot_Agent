import re
import uuid
from datetime import date as date_cls

from dotenv import load_dotenv

load_dotenv()  # must run before graph/stt/extraction modules read os.environ

import streamlit as st
from langgraph.types import Command

from graph import build_graph
from pdf_export import build_pdf_html, html_to_pdf_bytes
from schema import MedicineEntry
from ui_style import inject_base_styles, letterhead, section_label, render_prescription_sheet

DOCTOR_NAME = "xyz"  # fixed per requirement — not user-editable

st.set_page_config(page_title="Voice Rx", layout="wide", page_icon="℞")
inject_base_styles()
letterhead("Voice-to-Prescription", "Dictate. Review. Confirm. Every correction is logged.")


def _strength_missing_unit(strength: str | None) -> bool:
    """
    True if a strength was captured but has no unit attached (e.g. '625'
    instead of '625 mg'). We deliberately never auto-fill a guessed unit —
    silently assuming mg vs mcg vs ml is a real harm vector — we only flag
    it so the doctor supplies the unit explicitly during review.
    """
    if not strength:
        return False
    return not re.search(r"(mg|ml|mcg|g|%)\b", strength, flags=re.IGNORECASE)


if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pending_review" not in st.session_state:
    st.session_state.pending_review = None
if "review_round" not in st.session_state:
    st.session_state.review_round = 0
# patient fields captured once and held here explicitly, independent of
# whether the input widgets are still on screen — avoids losing them
# once the flow moves past the first stage
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {"name": "", "age": "", "gender": ""}

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Patient fields + dictation (first stage only) ---
if st.session_state.pending_review is None and not st.session_state.get("final_prescription"):
    hcol1, hcol2, hcol3 = st.columns(3)
    patient_name = hcol1.text_input("Patient name", value=st.session_state.patient_info["name"])
    patient_age = hcol2.text_input("Age", value=st.session_state.patient_info["age"])
    patient_gender = hcol3.selectbox(
        "Gender",
        ["", "Male", "Female", "Other"],
        index=["", "Male", "Female", "Other"].index(st.session_state.patient_info["gender"] or ""),
    )
    # captured into session_state immediately, not just left in the widgets,
    # so it survives once this block stops rendering after the button click
    st.session_state.patient_info = {"name": patient_name, "age": patient_age, "gender": patient_gender}

    section_label("DICTATION")
    audio = st.audio_input("Dictate the prescription")

    if audio is not None:
        if st.button("Transcribe & Extract", type="primary"):
            with st.spinner("Transcribing and extracting medicines..."):
                result = st.session_state.graph.invoke(
                    {
                        "audio_bytes": audio.getvalue(),
                        "audio_filename": "input.wav",
                        "doctor_edits": None,
                    },
                    config=config,
                )
            interrupt_payload = result["__interrupt__"][0].value
            st.session_state.pending_review = interrupt_payload["medicines_for_review"]
            st.session_state.review_round += 1
            st.rerun()

# --- Stage 2: HITL review screen ---
if st.session_state.pending_review is not None:
    section_label("REVIEW BEFORE CONFIRMING")
    st.caption(
        "Amber rows are below 88% match confidence, or have a strength with no unit — "
        "verify these manually before finalizing."
    )

    with st.expander("Not right? Re-record instead of editing"):
        redo_audio = st.audio_input("Re-dictate this prescription", key="redo_audio")
        if redo_audio is not None and st.button("Re-transcribe"):
            st.session_state.thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Re-transcribing and extracting..."):
                result = st.session_state.graph.invoke(
                    {
                        "audio_bytes": redo_audio.getvalue(),
                        "audio_filename": "input.wav",
                        "doctor_edits": None,
                    },
                    config=config,
                )
            interrupt_payload = result["__interrupt__"][0].value
            st.session_state.pending_review = interrupt_payload["medicines_for_review"]
            st.session_state.review_round += 1
            st.rerun()

    edited_rows = []
    for i, med in enumerate(st.session_state.pending_review):
        entry = MedicineEntry(**med)
        name_flagged = entry.match_confidence < 0.88
        unit_flagged = _strength_missing_unit(entry.strength)
        row_flagged = name_flagged or unit_flagged

        row_class = "rx-row-flag" if row_flagged else "rx-row-ok"
        st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)

        cols = st.columns([3, 2, 2, 2, 2])
        name_label = f"Drug name (heard: {entry.raw_text})" if name_flagged else "Drug name"
        drug_name = cols[0].text_input(name_label, entry.drug_name, key=f"drug_{st.session_state.review_round}_{i}")
        strength_label = "Strength (add unit)" if unit_flagged else "Strength"
        strength = cols[1].text_input(strength_label, entry.strength or "", key=f"strength_{st.session_state.review_round}_{i}")
        frequency = cols[2].text_input("Frequency", entry.frequency or "", key=f"freq_{st.session_state.review_round}_{i}")
        duration = cols[3].text_input("Duration", entry.duration or "", key=f"dur_{st.session_state.review_round}_{i}")
        instructions = cols[4].text_input("Instructions", entry.instructions or "", key=f"instr_{st.session_state.review_round}_{i}")

        st.markdown(
            f'<div class="rx-match-note">Match: {entry.match_status} — confidence {entry.match_confidence:.0%}</div>',
            unsafe_allow_html=True,
        )
        if unit_flagged:
            st.markdown(
                f'<div class="rx-flag-note">No unit was spoken for the strength of '
                f"{entry.drug_name} ('{entry.strength}') — confirm mg/ml/mcg/etc.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        entry.drug_name = drug_name
        entry.strength = strength or None
        entry.frequency = frequency or None
        entry.duration = duration or None
        entry.instructions = instructions or None
        entry.confirmed_by_doctor = True
        edited_rows.append(entry.model_dump())

    if st.button("Confirm & Finalize Prescription", type="primary"):
        with st.spinner("Finalizing..."):
            final = st.session_state.graph.invoke(
                Command(resume={"medicines": edited_rows}), config=config
            )
        st.session_state.pending_review = None
        st.session_state.final_prescription = final["medicines"]
        st.session_state.rx_date = date_cls.today()
        st.rerun()

# --- Stage 3: final output — rendered as an actual prescription sheet ---
if st.session_state.get("final_prescription"):
    section_label("FINALIZED PRESCRIPTION")

    patient = st.session_state.patient_info
    patient_display = patient["name"]
    if patient["age"] or patient["gender"]:
        patient_display += f" ({patient['age']}{', ' if patient['age'] and patient['gender'] else ''}{patient['gender']})"

    sheet_html = render_prescription_sheet(
        patient_name=patient_display,
        doctor_name=DOCTOR_NAME,
        date=str(st.session_state.get("rx_date", "")),
        medicines=st.session_state.final_prescription,
    )
    st.markdown(sheet_html, unsafe_allow_html=True)
    st.write("")

    pdf_html = build_pdf_html(
        patient_name=patient["name"],
        age=patient["age"],
        gender=patient["gender"],
        doctor_name=DOCTOR_NAME,
        date=str(st.session_state.get("rx_date", "")),
        medicines=st.session_state.final_prescription,
    )
    pdf_bytes = html_to_pdf_bytes(pdf_html)

    dl_col, restart_col = st.columns([1, 1])
    dl_col.download_button(
        "Download as PDF",
        data=pdf_bytes,
        file_name=f"prescription_{patient['name'] or 'patient'}_{date_cls.today()}.pdf",
        mime="application/pdf",
    )
    if restart_col.button("Start New Prescription"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_review = None
        st.session_state.final_prescription = None
        st.session_state.patient_info = {"name": "", "age": "", "gender": ""}
        st.rerun()