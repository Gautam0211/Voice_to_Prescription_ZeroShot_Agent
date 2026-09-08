"""
Extraction node — structured NER-style extraction, not generation.

Deliberately "dumb and literal": the model's only job is to pull out what
was said, split into fields. It must NOT try to correct or normalize drug
names — that's the correction layer's job (correction.py), kept separate
so every correction is auditable and attributable to a specific algorithm,
not buried inside a single opaque LLM call.

Uses Groq's gpt-oss-20b by default — fast and cheap, sufficient for
structured extraction. gpt-oss-120b is available as a fallback for
transcripts that fail to parse cleanly (see extract_with_fallback).
"""

import json
import os

from groq import Groq
from pydantic import ValidationError

from schema import MedicineEntry

_client: Groq | None = None

EXTRACTION_SYSTEM_PROMPT = """You are a literal transcription-to-structure extractor for \
spoken medical prescriptions. Given a raw transcript, extract EACH medicine mentioned as \
a separate object.

CRITICAL: "raw_text" and "drug_name" must contain ONLY the drug name itself — never the \
strength, dose, frequency, duration, or any other field. Those go in their own separate \
fields. Do NOT correct, normalize, or guess at "proper" drug spellings — copy the drug \
name exactly as it sounds in the transcript. Correction happens in a later, separate step.
If a field wasn't mentioned, use null.

Generic category words like "cough syrup", "tablet", "syrup", or "medicine" must NEVER be extracted as their own separate medicine entry. If such a word is mentioned alongside a specific drug/brand name (e.g. "Ascoril cough syrup"), attach it only as part of that single entry's raw_text/drug_name — do not create an additional entry for the category word. If no drug/brand is mentioned/named with word "cough syrup", only and only then add it as an individual log of "cough syrup".

Example:
Input: "azithromycin 500 mg once daily for 3 days"
Output: {"medicines": [{"raw_text": "azithromycin", "drug_name": "azithromycin", \
"strength": "500 mg", "dosage_form": null, "frequency": "once daily", "duration": "3 days", \
"route": null, "instructions": null}]}

Return ONLY valid JSON in this exact shape, no prose, no markdown fences:
{
  "medicines": [
    {
      "raw_text": "...",
      "drug_name": "...",
      "strength": "... or null",
      "dosage_form": "... or null",
      "frequency": "... or null",
      "duration": "... or null",
      "route": "... or null",
      "instructions": "... or null"
    }
  ]
}"""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        _client = Groq(api_key=api_key)
    return _client


def _call_model(transcript: str, model: str) -> dict:
    client = get_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
        temperature=0,  # deterministic extraction, not creative generation
    )
    return json.loads(completion.choices[0].message.content)


def extract_medicines(transcript: str) -> list[MedicineEntry]:
    """
    Extracts medicines from a transcript. Tries the fast model first;
    falls back to the larger model only if the fast model's output fails
    schema validation (keeps the common case cheap and fast, per Groq's
    performance advantage, while still being robust on hard inputs).
    """
    for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        try:
            raw = _call_model(transcript, model)
            entries = [MedicineEntry(**m) for m in raw.get("medicines", [])]
            return entries
        except (ValidationError, json.JSONDecodeError, KeyError):
            continue  # try the next (bigger) model

    raise RuntimeError(
        "Extraction failed on both models — transcript may be too garbled. "
        "Surface this to the doctor for manual entry rather than silently failing."
    )