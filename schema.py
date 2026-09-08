"""
Core data schema for the voice-to-prescription agent.
Every node (extraction, correction, HITL, output) reads/writes this shape.
Keeping this in one place means a schema change never requires touching
node logic in more than one file.
"""

from pydantic import BaseModel, Field
from typing import Optional


class MedicineEntry(BaseModel):
    """One medicine line inside a prescription."""

    raw_text: str = Field(
        description="Exact text as extracted from the transcript, before correction"
    )
    drug_name: str = Field(description="Best-known drug name after correction")
    strength: Optional[str] = Field(
        default=None, description="e.g. '500mg', '5ml'"
    )
    dosage_form: Optional[str] = Field(
        default=None, description="tablet / syrup / injection / capsule / drops"
    )
    frequency: Optional[str] = Field(
        default=None, description="e.g. 'twice daily', 'TDS', 'BD'"
    )
    duration: Optional[str] = Field(
        default=None, description="e.g. '5 days', '2 weeks'"
    )
    route: Optional[str] = Field(
        default=None, description="oral / topical / IV / IM"
    )
    instructions: Optional[str] = Field(
        default=None, description="e.g. 'after food', 'before sleep'"
    )

    # correction metadata — this is what a founder will want to see exists
    match_status: str = Field(
        default="unverified",
        description="one of: exact_match, fuzzy_match, corrected, unmatched, unverified",
    )
    match_confidence: float = Field(
        default=0.0, description="0.0-1.0 confidence score from the correction layer"
    )
    confirmed_by_doctor: bool = Field(
        default=False, description="True only after explicit HITL confirmation"
    )


class Prescription(BaseModel):
    """Full prescription — the final unit that gets rendered and persisted."""

    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    medicines: list[MedicineEntry] = Field(default_factory=list)
    raw_transcript: Optional[str] = Field(
        default=None, description="Full unedited STT output, kept for audit/logging"
    )
    finalized: bool = Field(
        default=False, description="True only once every medicine is doctor-confirmed"
    )
