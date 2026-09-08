"""
Graph wiring. State flows: audio -> transcript -> extracted medicines ->
corrected medicines -> [PAUSE for doctor] -> finalized prescription.

The HITL step uses LangGraph's `interrupt()` — the graph genuinely halts
execution and returns control to Streamlit; it resumes only when the app
calls the graph again with a Command(resume=...). This needs a
checkpointer to persist state across that pause. MemorySaver is fine for
a single-session demo; swap for a SQLite/Postgres checkpointer if you
need state to survive an app restart (Streamlit Cloud restarts idle apps).
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from correction import correct_drug_name, load_drug_list
from extraction import extract_medicines
from schema import MedicineEntry, Prescription
from stt import transcribe_audio


class GraphState(TypedDict):
    audio_bytes: bytes
    audio_filename: str
    transcript: str
    medicines: list[dict]  # MedicineEntry dumped to dict (LangGraph state must be serializable)
    doctor_edits: dict | None  # populated on resume, from the HITL screen


def stt_node(state: GraphState) -> dict:
    drug_list = load_drug_list()
    transcript = transcribe_audio(
        state["audio_bytes"], state["audio_filename"], drug_list=drug_list
    )
    return {"transcript": transcript}


def extraction_node(state: GraphState) -> dict:
    entries = extract_medicines(state["transcript"])
    return {"medicines": [e.model_dump() for e in entries]}


def correction_node(state: GraphState) -> dict:
    drug_list = load_drug_list()
    corrected = []
    for m in state["medicines"]:
        entry = MedicineEntry(**m)
        result = correct_drug_name(entry.raw_text, drug_list)
        if result.status != "unmatched":
            entry.drug_name = result.matched_name
        entry.match_status = result.status
        entry.match_confidence = result.confidence
        corrected.append(entry.model_dump())
    return {"medicines": corrected}


def hitl_node(state: GraphState) -> dict:
    """
    Pauses the graph and hands the current medicine list to Streamlit for
    doctor review. `interrupt()` returns whatever value the app passes in
    when it resumes with Command(resume=...) — we expect a dict of
    doctor-edited medicines.
    """
    doctor_edits = interrupt({"medicines_for_review": state["medicines"]})
    return {"medicines": doctor_edits["medicines"]}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("stt", stt_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("correction", correction_node)
    graph.add_node("hitl", hitl_node)

    graph.add_edge(START, "stt")
    graph.add_edge("stt", "extraction")
    graph.add_edge("extraction", "correction")
    graph.add_edge("correction", "hitl")
    graph.add_edge("hitl", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
