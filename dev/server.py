#!/usr/bin/env python3
"""FastAPI server for RAG-HPO pipeline.

Start with:
    uvicorn dev.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uuid
import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from config import settings
    from schemas import (
        ExtractRequest,
        ExtractResponse,
        JobStatus,
        HealthResponse,
        ResultItem,
    )
except ImportError:
    from dev.config import settings
    from dev.schemas import (
        ExtractRequest,
        ExtractResponse,
        JobStatus,
        HealthResponse,
        ResultItem,
    )

# ── Globals ────────────────────────────────────────────────────────

_pipeline = None
_jobs: dict[str, dict] = {}  # job_id → {status, results, error, ...}
_lock = threading.Lock()

# ── Lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _pipeline

    try:
        from rag_hpo import LLMClient
        from pipeline import Pipeline
    except ImportError:
        from dev.rag_hpo import LLMClient
        from dev.pipeline import Pipeline

    # Build LLM client from settings
    llm = LLMClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_name=settings.model,
        max_tokens_per_day=settings.max_tokens_per_day,
        max_queries_per_minute=settings.max_queries_per_minute,
        temperature=settings.temperature,
    )
    print(f"LLM client initialized: {settings.model} via {settings.base_url}")

    _pipeline = Pipeline()
    _pipeline.initialize(llm)
    print("Pipeline ready — models loaded, FAISS index built.")

    yield

    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG-HPO",
    description="HPO term extraction from clinical notes via LLM + RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None and _pipeline.is_ready,
        queue_size=sum(1 for j in _jobs.values() if j["status"] == "processing"),
    )


@app.post("/api/v1/extract", response_model=ExtractResponse, status_code=202)
async def extract(req: ExtractRequest):
    """Submit clinical notes for HPO extraction. Returns a job_id to poll."""
    if _pipeline is None or not _pipeline.is_ready:
        raise HTTPException(503, "Pipeline not initialized yet. Try again shortly.")

    job_id = str(uuid.uuid4())[:8]
    notes = [n.model_dump() for n in req.notes]

    with _lock:
        _jobs[job_id] = {
            "status": "processing",
            "notes_count": len(notes),
            "results": None,
            "elapsed_seconds": None,
            "error": None,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, notes), daemon=True)
    thread.start()

    return ExtractResponse(job_id=job_id, status="processing", notes_count=len(notes))


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status and (if done) results of an extraction job."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return JobStatus(job_id=job_id, **job)


# ── Helpers ────────────────────────────────────────────────────────


def _run_job(job_id: str, notes: list[dict]):
    """Process a batch of notes in a background thread."""
    t0 = time.time()
    try:
        results = _pipeline.run(notes)
        items = [ResultItem(**r) for r in results]
    except Exception as exc:
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
        return

    elapsed = time.time() - t0
    with _lock:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["results"] = [i.model_dump() for i in items]
        _jobs[job_id]["elapsed_seconds"] = round(elapsed, 1)
