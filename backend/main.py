"""Unclump API — FastAPI server for the ADHD task breakdown service."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from engine import breakdown_task, breakdown_task_fallback, TaskBreakdown

app = FastAPI(
    title="Unclump API",
    description="AI-powered task breakdown for ADHD brains. Turns vague tasks into micro-steps so small you can't fail to start.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The task you're stuck on. Be as messy as you like.",
        examples=["clean my room", "reply to that email from last week", "start my taxes"],
    )


class TaskResponse(BaseModel):
    original_task: str
    entry_hook: str
    total_estimated_minutes: float
    micro_steps: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "unclump"}


@app.post("/api/breakdown", response_model=TaskResponse)
async def breakdown(request: TaskRequest):
    """Break a task into ADHD-friendly micro-steps.

    Uses OpenAI when available, falls back to template-based breakdown
    if the API key is not set or the call fails.
    """
    task = request.task.strip()

    # Try AI first, fall back to templates
    if os.getenv("OPENAI_API_KEY"):
        try:
            result = breakdown_task(task)
            return result.to_dict()
        except Exception as e:
            # Log the error but don't expose details to the client
            print(f"AI breakdown failed, using fallback: {e}")
            result = breakdown_task_fallback(task)
            return result.to_dict()
    else:
        result = breakdown_task_fallback(task)
        return result.to_dict()


@app.get("/")
async def root():
    return {
        "name": "Unclump API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoint": "POST /api/breakdown",
    }


# ── Simple email waitlist (no DB needed for MVP) ──

waitlist_emails: list[str] = []


class WaitlistRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)


@app.post("/api/waitlist")
async def waitlist(request: WaitlistRequest):
    email = request.email.strip().lower()
    if email not in waitlist_emails:
        waitlist_emails.append(email)
        print(f"[WAITLIST] New signup: {email} (total: {len(waitlist_emails)})")
    return {"status": "ok", "message": "You're on the list!"}
