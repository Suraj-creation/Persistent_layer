from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatasetAuditReport(BaseModel):
    """Suitability report generated during dataset intake."""
    dataset_name: str
    total_records: int
    total_students: int
    total_items: int
    total_concepts: int
    text_availability_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of interactions with free-text responses")
    misconception_label_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of errors with verified misconception tags")
    missingness_by_column: Dict[str, float]
    concept_distribution: Dict[str, int]
    leakage_risk_flags: List[str]
    suitability_verdict: str  # e.g., "SUITABLE_FOR_LANGUAGE_GROUNDED_MISCONCEPTION_MODELING"
    notes: List[str]
