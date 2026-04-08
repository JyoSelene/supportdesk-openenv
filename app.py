"""
FastAPI server exposing the SupportDesk-OpenEnv HTTP API.

Endpoints
---------
GET  /              — Health check / welcome
GET  /health        — Liveness probe (returns {"status": "ok"})
GET  /tasks         — List available tasks with metadata
POST /reset         — Start a new episode (returns observation)
POST /step          — Submit an action (returns obs, reward, done, info)
GET  /state         — Return current environment state
GET  /openenv.yaml  — Serve the openenv.yaml spec file

Session management
------------------
Each call to /reset creates a new session_id (UUID). The session_id must be
passed to /step and /state. Up to MAX_SESSIONS sessions are kept in memory;
oldest sessions are evicted when the limit is reached.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from email_triage_env import SupportDeskEnv
from email_triage_env.data import TASK_CONFIG
from email_triage_env.models import (
    EmailAction,
    EmailObservation,
    EnvironmentState,
    ResetRequest,
    StepRequest,
    StepResponse,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SupportDesk-OpenEnv",
    description=(
        "Customer email triage and response environment for AI agent training. "
        "Implements the OpenEnv spec with 3 tasks: email_classify (easy), "
        "email_extract (medium), email_respond (hard)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

MAX_SESSIONS = 50  # Keep memory bounded on small HF Spaces instances

class SessionStore:
    """Simple in-memory LRU session store."""

    def __init__(self, max_size: int = MAX_SESSIONS) -> None:
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # evict oldest
        self._store[session_id] = {
            "env": SupportDeskEnv(seed=42),
            "created_at": time.time(),
            "last_used": time.time(),
        }
        return session_id

    def get(self, session_id: str) -> Optional[SupportDeskEnv]:
        entry = self._store.get(session_id)
        if entry is None:
            return None
        entry["last_used"] = time.time()
        self._store.move_to_end(session_id)
        return entry["env"]

    def __len__(self) -> int:
        return len(self._store)


sessions = SessionStore()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "SupportDesk-OpenEnv",
        "version": "1.0.0",
        "description": "Customer email triage environment — OpenEnv compliant",
        "tasks": list(TASK_CONFIG.keys()),
        "docs": "/docs",
        "openenv_spec": "/openenv.yaml",
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/tasks", tags=["environment"])
async def list_tasks():
    """List all available tasks with difficulty, description, and step count."""
    return {
        task_name: {
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "num_emails": len(cfg["emails"]),
        }
        for task_name, cfg in TASK_CONFIG.items()
    }


@app.post("/reset", response_model=Dict[str, Any], tags=["environment"])
async def reset(request: ResetRequest):
    """
    Start a new episode.

    Returns the session_id and the first observation.
    Pass session_id to all subsequent /step and /state calls.
    """
    task_name = request.task_name
    if task_name not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG.keys())}",
        )

    session_id = sessions.create()
    env = sessions.get(session_id)
    assert env is not None

    try:
        obs = env.reset(task_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(request: StepRequest):
    """
    Submit an action and receive the next observation.

    Returns:
    - observation (null when done=True)
    - reward [0.0, 1.0]
    - done (bool)
    - info (grader feedback + episode metadata)
    """
    env = sessions.get(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /reset first.",
        )

    try:
        next_obs, reward, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StepResponse(
        observation=next_obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EnvironmentState, tags=["environment"])
async def state(session_id: str):
    """Return the full internal state of the environment for this session."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return env.state()


@app.get("/openenv.yaml", tags=["meta"])
async def serve_openenv_yaml():
    """Serve the openenv.yaml spec file."""
    yaml_path = Path(__file__).parent / "openenv.yaml"
    if yaml_path.exists():
        return FileResponse(str(yaml_path), media_type="text/yaml")
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
