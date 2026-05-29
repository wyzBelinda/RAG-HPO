"""Pydantic models for RAG-HPO FastAPI service."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class NoteItem(BaseModel):
    """A single clinical note to process."""

    patient_id: str = Field(..., description="Unique patient identifier")
    clinical_note: str = Field(..., description="Free-text clinical note")


class ExtractRequest(BaseModel):
    """Request body for the extract endpoint."""

    notes: list[NoteItem] = Field(..., min_length=1, max_length=500)


class ResultItem(BaseModel):
    """A single mapped HPO term result."""

    patient_id: str
    phrase: str
    category: str
    hpo_id: Optional[str] = None


class ExtractResponse(BaseModel):
    """Response returned immediately after submitting an extraction job."""

    job_id: str
    status: str = "processing"
    notes_count: int


class JobStatus(BaseModel):
    """Status and results of an extraction job."""

    job_id: str
    status: str  # "processing" | "done" | "failed"
    notes_count: int
    results: Optional[list[ResultItem]] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    model_loaded: bool
    queue_size: int
