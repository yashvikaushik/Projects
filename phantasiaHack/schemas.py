"""
schemas.py
----------
Pydantic data models for structured symptom extraction output.
Designed for extensibility into downstream stages (e.g., risk scoring, triage).
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedSymptom(BaseModel):
    """
    Represents a single symptom extracted from clinical text.

    Fields are intentionally kept broad so the model can be extended
    without breaking existing consumers.
    """

    name: str = Field(
        ...,
        description="Normalized symptom name (e.g., 'lower back pain').",
    )
    severity: Optional[str] = Field(
        None,
        description="Pain/severity qualifier detected near the symptom token "
                    "(e.g., 'sharp', 'dull', 'severe', 'mild').",
    )
    location: Optional[str] = Field(
        None,
        description="Anatomical location associated with the symptom "
                    "(e.g., 'right lower back', 'chest', 'abdomen').",
    )
    duration: Optional[str] = Field(
        None,
        description="Temporal phrase describing how long the symptom has been present "
                    "(e.g., '2 days', 'since last week', 'for three hours').",
    )
    negated: bool = Field(
        False,
        description="True if the symptom was preceded by a negation cue "
                    "(e.g., 'no fever', 'denies chest pain').",
    )
    raw_text: Optional[str] = Field(
        None,
        description="Original span text from the input as a provenance reference.",
    )


class ExtractionResult(BaseModel):
    """
    Top-level output of the symptom extraction pipeline.

    Separates active symptoms from negated ones for clarity.
    """

    symptoms: List[ExtractedSymptom] = Field(
        default_factory=list,
        description="Symptoms confirmed as present (negated=False).",
    )
    negated_symptoms: List[str] = Field(
        default_factory=list,
        description="Names of symptoms explicitly negated in the source text.",
    )
    input_text: str = Field(
        ...,
        description="Original input string, preserved for audit/debugging.",
    )
