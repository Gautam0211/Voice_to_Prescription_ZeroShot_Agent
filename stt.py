"""
Speech-to-text layer — Groq's hosted Whisper-large-v3.

Why Groq for STT specifically: Groq's LPU inference is dramatically faster
than CPU-bound local Whisper, which matters a lot on Streamlit Cloud since
the free/community tier has no GPU and limited CPU — running Whisper
locally there would be slow and likely to hit resource limits. Using
Groq's API keeps the Streamlit app itself lightweight; it's just making
HTTP calls, not loading a model into memory.

`initial_prompt` biasing: Whisper (and Groq's hosted version) accepts a
prompt hint that nudges decoding toward specific vocabulary. We pass a
sample of common drug names so the decoder is more likely to output them
correctly, per Phase 1 of the roadmap. This does NOT replace the
correction layer — it just reduces how many errors reach it.
"""

import os
from groq import Groq

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env locally or to "
                "Streamlit Cloud's secrets.toml before deploying."
            )
        _client = Groq(api_key=api_key)
    return _client


def build_vocab_prompt(drug_list: list[str], sample_size: int = 40) -> str:
    """
    Whisper's initial_prompt has a length limit and diminishing returns
    past a few dozen terms, so we sample rather than dump the full list.
    """
    sample = drug_list[:sample_size]
    return "Common medicine names: " + ", ".join(sample) + "."


def transcribe_audio(audio_bytes: bytes, filename: str, drug_list: list[str] | None = None) -> str:
    """
    Sends raw audio bytes to Groq Whisper and returns the raw transcript.
    filename must carry a real extension (e.g. 'input.wav') — Groq's API
    uses it to infer the audio format.
    """
    client = get_client()
    prompt = build_vocab_prompt(drug_list) if drug_list else None

    response = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        prompt=prompt,
        response_format="text",
        language="en",
    )
    # response is a plain string when response_format="text"
    return response if isinstance(response, str) else response.text
