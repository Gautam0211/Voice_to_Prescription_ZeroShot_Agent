"""
Correction / grounding layer.

Takes a raw (possibly misheard) drug name from STT and grounds it against
a known drug reference list using two complementary signals:

1. Character-level similarity (rapidfuzz) — catches spelling-adjacent errors.
2. Phonetic similarity (jellyfish's metaphone) — catches sound-alike errors,
   which is what STT actually produces most of the time (it mishears sounds,
   not spellings). This is the piece that catches "alhophosebia" ->
   "thalassophobia" style errors that pure edit-distance often misses.

Design choice worth calling out in an interview: this is 100% zero-shot.
No model is trained or fine-tuned. We compose pretrained STT + classic
string algorithms + an optional LLM disambiguation call. This keeps the
system cheap, fast, and auditable — every correction can be logged with
the exact score that produced it.
"""

import csv
import functools
import re
from dataclasses import dataclass

import jellyfish
from rapidfuzz import fuzz, process


def _strip_non_name_tokens(text: str) -> str:
    """
    Defense-in-depth: even if the extraction node leaks strength/frequency
    words into the name field, strip common trailing dose/unit/frequency
    tokens before fuzzy matching so the correction layer isn't penalized
    for someone else's extraction bug.
    """
    text = re.sub(
        r"\b\d+(\.\d+)?\s*(mg|ml|mcg|g|tablet[s]?|capsule[s]?|drop[s]?)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(once|twice|thrice|\d+\s*times?)\s+(a\s+)?(day|daily)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bfor\s+\d*\s*(day|days|week|weeks)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(bd|tds|od|qid|sos)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(tablet[s]?|capsule[s]?|drop[s]?|dose[s]?)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "", text)  # stray bare numbers, stripped last
    return re.sub(r"\s+", " ", text).strip(" ,.")


@dataclass
class MatchResult:
    matched_name: str | None
    status: str  # exact_match | fuzzy_match | unmatched
    confidence: float  # 0.0 - 1.0


# Thresholds — tuned starting points, log real corrections and adjust later.
EXACT_THRESHOLD = 100
HIGH_CONFIDENCE_THRESHOLD = 88
LOW_CONFIDENCE_FLOOR = 65  # below this, don't even suggest a match


@functools.lru_cache(maxsize=1)
def load_drug_list(csv_path: str = "data/drugs.csv") -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["drug_name"] for row in reader]


def _phonetic_score(raw: str, candidate: str) -> float:
    """
    Returns 0-100 phonetic similarity using metaphone codes.
    Two words with identical metaphone codes get 100; otherwise we fall
    back to char-similarity between the two metaphone codes themselves,
    which softly rewards "close but not identical" sounds.
    """
    m1, m2 = jellyfish.metaphone(raw), jellyfish.metaphone(candidate)
    if not m1 or not m2:
        return 0.0
    if m1 == m2:
        return 100.0
    return fuzz.ratio(m1, m2)


def correct_drug_name(raw_name: str, drug_list: list[str] | None = None) -> MatchResult:
    """
    Ground a single raw drug name against the reference list.
    Combines rapidfuzz's token-based score with a phonetic score, taking
    the max — either signal firing strongly is enough to suggest a match.
    """
    drug_list = drug_list or load_drug_list()
    raw_clean = _strip_non_name_tokens(raw_name.strip())

    # 1. Fast path: exact case-insensitive match
    for candidate in drug_list:
        if candidate.lower() == raw_clean.lower():
            return MatchResult(matched_name=candidate, status="exact_match", confidence=1.0)

    # 2. Character-level fuzzy match (best candidate + score)
    fuzzy_best = process.extractOne(raw_clean, drug_list, scorer=fuzz.WRatio)
    fuzzy_candidate, fuzzy_score, _ = fuzzy_best if fuzzy_best else (None, 0, None)

    # 3. Phonetic match — score every candidate, take the best
    phonetic_scores = [(c, _phonetic_score(raw_clean, c)) for c in drug_list]
    phonetic_candidate, phonetic_score = max(phonetic_scores, key=lambda x: x[1])

    # 4. Combine — take whichever signal is more confident
    if fuzzy_score >= phonetic_score:
        best_candidate, best_score = fuzzy_candidate, fuzzy_score
    else:
        best_candidate, best_score = phonetic_candidate, phonetic_score

    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return MatchResult(
            matched_name=best_candidate, status="fuzzy_match", confidence=best_score / 100
        )
    if best_score >= LOW_CONFIDENCE_FLOOR:
        return MatchResult(
            matched_name=best_candidate, status="fuzzy_match", confidence=best_score / 100
        )

    return MatchResult(matched_name=None, status="unmatched", confidence=best_score / 100)


def top_candidates(raw_name: str, drug_list: list[str] | None = None, n: int = 3) -> list[str]:
    """
    Returns the top-N closest drug names for a raw input — used to feed
    the LLM-corrector pass (Phase 3, step 3) or to populate a dropdown
    in the HITL review screen so the doctor can pick instead of retype.
    """
    drug_list = drug_list or load_drug_list()
    results = process.extract(raw_name, drug_list, scorer=fuzz.WRatio, limit=n)
    return [r[0] for r in results]