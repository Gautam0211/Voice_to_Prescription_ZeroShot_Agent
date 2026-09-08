# Voice-to-Prescription Agent

A doctor dictates a prescription out loud → the system transcribes it, extracts structured medicine data, corrects misheard drug names, pauses for the doctor to review and confirm (human-in-the-loop), then generates a finalized, downloadable prescription — as a clinical-styled web page and a PDF.

Built as a [LangGraph](https://github.com/langchain-ai/langgraph) agent on top of [Groq](https://groq.com) (Whisper-large-v3 for speech-to-text, gpt-oss-20b/120b for structured extraction), with a [Streamlit](https://streamlit.io) front end.

## Why this exists

Voice input for a blog-writing agent is a "nice to have." Voice input for a doctor is a **safety-critical** problem: a misheard drug name or dose isn't a quality issue, it's a patient-harm issue. This project is built around that distinction — every design decision here optimizes for catching and surfacing uncertainty, not hiding it.

## Architecture

```
audio_input
    ↓
[STT node]           Groq Whisper-large-v3, with drug-vocabulary biasing
    ↓
[extraction node]    Groq gpt-oss-20b (fallback: 120b) — literal, structured
    ↓                extraction only. Never corrects or guesses.
[correction node]    Zero-shot grounding: fuzzy match (rapidfuzz) +
    ↓                phonetic match (jellyfish) against a drug reference
    ↓                list. No fine-tuning — pretrained STT + pretrained LLM
    ↓                + classic string algorithms, composed.
[HITL interrupt]     Graph genuinely pauses (LangGraph interrupt()) and
    ↓                hands the medicine list to the doctor for review.
    ↓                Low-confidence matches and strengths with no unit are
    ↓                flagged — never silently auto-corrected.
[output]             Rendered as a prescription sheet (web + PDF)
```

## Why this is zero-shot

No model here is trained or fine-tuned. The correction layer composes:
- Off-the-shelf pretrained Whisper (biased with a vocabulary prompt, not fine-tuned)
- Classic string-distance algorithms (Levenshtein/WRatio via `rapidfuzz`, phonetic matching via `jellyfish`'s metaphone)
- An LLM used for structured extraction, not for "knowing" drug names

This keeps the system cheap, fast, auditable, and easy to extend — every correction can be traced back to the exact score that produced it.

## Why human-in-the-loop is mandatory, not optional

Two real drug names can be dangerously close to each other (e.g. Hydroxyzine / Hydralazine). No amount of fuzzy-matching eliminates that ambiguity completely — a match score of 100% only means "closest in *this* reference list," not "definitely correct." The system never finalizes a prescription without explicit doctor confirmation, and it never silently fills in a missing unit (e.g. assuming "625" means "625 mg") — both are logged as flags for the doctor to resolve, not auto-resolved.

## Project structure

```
schema.py        Pydantic models — the shared contract every node reads/writes
stt.py            Speech-to-text (Groq Whisper), with vocabulary biasing
extraction.py     Raw transcript → structured, unverified medicine list
correction.py     Fuzzy + phonetic grounding against data/drugs.csv
data/drugs.csv    Reference drug list (generics + common brand names)
graph.py          LangGraph wiring: STT → extraction → correction → HITL
app.py            Streamlit UI — capture, review screen, final output
ui_style.py       Clinical letterhead styling (CSS + HTML renderers)
pdf_export.py     PDF generation via xhtml2pdf, for the download button
test.py           Standalone script to sanity-check your Groq API key
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env          # then add your real GROQ_API_KEY
```

Verify your key works before anything else:

```bash
python test.py
```

Run locally:

```bash
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Connect it on [Streamlit Cloud](https://streamlit.io/cloud).
3. Add your key under **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   Streamlit Cloud does **not** read `.env` files — secrets must be set here.

Groq's hosted inference is the reason this runs comfortably on Streamlit Cloud's free tier: no local model to load, no GPU needed, just fast API calls.

## Known limitations / what's next

- `data/drugs.csv` is a curated sample list, not a full formulary (e.g. RxNorm) — swap it in for production use.
- No drug-drug interaction checking yet (e.g. flagging a dangerous combination like Warfarin + Aspirin after confirmation) — a natural next node in the graph.
- No persistence layer yet — finalized prescriptions aren't saved to a database, only downloadable as PDF per session.
- No authentication/digital signature layer.

## License

MIT (or update as needed).
