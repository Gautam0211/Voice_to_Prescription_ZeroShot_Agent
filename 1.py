"""
test_infra.py — independent test script for app.py's traced graph wrappers.
No other test files or fixtures required.

Run with:
    LANGSMITH_TRACING=false pytest test_infra.py -v
    or: python test_infra.py
"""
import sys
import types
from unittest.mock import MagicMock


class SessionStateStub(dict):
    """Mimics st.session_state: supports both dict-style and attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# --- Stub streamlit so app.py can be imported without a real Streamlit runtime ---
st_stub = types.ModuleType("streamlit")
for attr in ["set_page_config", "title", "audio_input", "button", "spinner",
             "rerun", "subheader", "caption", "container", "columns",
             "text_input", "table"]:
    setattr(st_stub, attr, MagicMock())
st_stub.session_state = SessionStateStub()
sys.modules["streamlit"] = st_stub

from app import run_transcription_and_extraction, run_finalize_prescription
from langgraph.types import Command


def test_transcription_invokes_graph_with_payload():
    graph = MagicMock()
    graph.invoke.return_value = {
        "__interrupt__": [MagicMock(value={"medicines_for_review": [{"id": 1}]})]
    }
    payload = {"audio_bytes": b"fake", "audio_filename": "input.wav", "doctor_edits": None}
    config = {"configurable": {"thread_id": "abc"}}

    result = run_transcription_and_extraction(graph, payload, config)

    graph.invoke.assert_called_once_with(payload, config=config)
    assert result["__interrupt__"][0].value["medicines_for_review"] == [{"id": 1}]


def test_finalize_invokes_graph_with_resume_command():
    graph = MagicMock()
    edited_rows = [{"drug_name": "Amoxicillin", "confirmed_by_doctor": True}]
    graph.invoke.return_value = {"medicines": edited_rows}
    config = {"configurable": {"thread_id": "abc"}}

    result = run_finalize_prescription(graph, edited_rows, config)

    call_args, call_kwargs = graph.invoke.call_args
    assert isinstance(call_args[0], Command)
    assert call_args[0].resume == {"medicines": edited_rows}
    assert call_kwargs["config"] == config
    assert result["medicines"] == edited_rows


def test_finalize_returns_graph_output_unmodified():
    graph = MagicMock()
    graph.invoke.return_value = {"medicines": [], "extra": "untouched"}

    result = run_finalize_prescription(graph, [], {})

    assert result == {"medicines": [], "extra": "untouched"}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))