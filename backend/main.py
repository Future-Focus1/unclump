"""Unclump API — FastAPI server for the ADHD task breakdown service."""

from datetime import datetime, timezone
import json as _json
import os
from pathlib import Path as _Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from engine import (
    breakdown_task,
    breakdown_task_fallback,
    create_adaptive_step,
    create_adaptive_step_fallback,
    create_coworking_messages,
    create_coworking_messages_fallback,
    create_support_message,
    create_support_message_fallback,
    create_unclump_session_plan,
    create_unclump_session_plan_fallback,
    has_ai_provider,
)

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


class SessionStartRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The task you're stuck on. Be as messy as you like.",
        examples=["I need to reply to an email but I keep avoiding it"],
    )
    energy_level: str | None = Field(
        default=None,
        max_length=50,
        description="Optional user-reported energy level, e.g. low, medium, high.",
    )
    preferred_tone: str | None = Field(
        default="gentle",
        max_length=50,
        description="Optional coaching tone, e.g. gentle, direct, playful.",
    )
    recent_friction: str | None = Field(
        default=None,
        max_length=300,
        description="Optional note about what made the task hard today.",
    )


class SessionFeedbackRequest(BaseModel):
    feedback: str = Field(
        ...,
        description="done, too_hard, need_smaller, distracted, not_right, or skip",
    )


class SessionResponse(BaseModel):
    session_id: str
    status: str
    original_task: str
    block_type: str
    block_label: str
    block_reason: str
    entry_hook: str
    reflection_prompt: str
    current_step_index: int
    completed_steps: int
    total_steps: int
    current_step: dict | None
    micro_steps: list[dict]
    feedback_options: list[str]
    nudge: str | None = None
    session_summary: str | None = None


class SupportRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=500)
    current_step: dict | None = None
    moment: str = Field(default="nudge", max_length=50)
    user_state: str | None = Field(default=None, max_length=300)


class SupportResponse(BaseModel):
    message: str
    suggested_action: str
    reminder_after_minutes: int
    tone: str


class CoworkerProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    task: str = Field(..., min_length=1, max_length=120)
    persona: str | None = Field(default=None, max_length=80)
    quirk: str | None = Field(default=None, max_length=40)


class CoworkingMessage(BaseModel):
    sender: str = Field(..., min_length=1, max_length=40)
    text: str = Field(..., min_length=1, max_length=240)


class CoworkingChatRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=500)
    mode: str = Field(default="periodic", max_length=30)
    session_minutes: int = Field(default=30, ge=10, le=60)
    coworkers: list[CoworkerProfile] = Field(default_factory=list, max_length=5)
    recent_messages: list[CoworkingMessage] = Field(default_factory=list, max_length=30)
    user_message: str | None = Field(default=None, max_length=500)


class CoworkingChatResponse(BaseModel):
    messages: list[CoworkingMessage]


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
    if has_ai_provider():
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


@app.post("/api/session/start", response_model=SessionResponse)
async def session_start(request: SessionStartRequest):
    """Start an adaptive task-rescue session."""
    task = request.task.strip()
    context = {
        "energy_level": request.energy_level,
        "preferred_tone": request.preferred_tone,
        "recent_friction": request.recent_friction,
    }

    if has_ai_provider():
        try:
            plan = create_unclump_session_plan(task, context=context)
        except Exception as e:
            print(f"AI session plan failed, using fallback: {e}")
            plan = create_unclump_session_plan_fallback(task, context=context)
    else:
        plan = create_unclump_session_plan_fallback(task, context=context)

    now = _now_iso()
    session = {
        **plan.to_dict(),
        "session_id": str(uuid4()),
        "status": "active",
        "current_step_index": 0,
        "completed_steps": 0,
        "feedback_counts": {},
        "feedback_log": [],
        "adaptation_log": [],
        "created_at": now,
        "updated_at": now,
        "nudge": None,
        "session_summary": None,
    }
    sessions = _load_sessions()
    sessions[session["session_id"]] = session
    _save_sessions(sessions)
    return _session_response(session)


@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def session_get(session_id: str):
    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_response(session)


@app.post("/api/session/{session_id}/feedback", response_model=SessionResponse)
async def session_feedback(session_id: str, request: SessionFeedbackRequest):
    """Advance or adapt the current session based on user feedback."""
    feedback = request.feedback.strip().lower()
    valid_feedback = {"done", "too_hard", "need_smaller", "distracted", "not_right", "skip"}
    if feedback not in valid_feedback:
        raise HTTPException(status_code=422, detail="Invalid feedback value")

    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("status") == "complete":
        return _session_response(session)

    steps = session.get("micro_steps", [])
    current_index = int(session.get("current_step_index", 0))
    current_step = steps[current_index] if current_index < len(steps) else None

    session["feedback_counts"][feedback] = session["feedback_counts"].get(feedback, 0) + 1
    session["feedback_log"].append({
        "feedback": feedback,
        "step_index": current_index,
        "at": _now_iso(),
    })

    if feedback == "done":
        session["completed_steps"] = int(session.get("completed_steps", 0)) + 1
        session["current_step_index"] = current_index + 1
        session["nudge"] = "Nice. The next step can stay small too."
    elif feedback == "skip":
        session["current_step_index"] = current_index + 1
        session["nudge"] = "Skipping is data, not failure. Keep moving to the next doorway."
    else:
        if not current_step:
            session["status"] = "complete"
            session["session_summary"] = _completion_summary(session)
        else:
            if has_ai_provider():
                try:
                    adapted = create_adaptive_step(
                        session["original_task"],
                        current_step,
                        feedback,
                        session["block_type"],
                    )
                except Exception as e:
                    print(f"AI adaptive step failed, using fallback: {e}")
                    adapted = create_adaptive_step_fallback(
                        session["original_task"],
                        current_step,
                        feedback,
                        session["block_type"],
                    )
            else:
                adapted = create_adaptive_step_fallback(
                    session["original_task"],
                    current_step,
                    feedback,
                    session["block_type"],
                )

            replacement = {
                **current_step,
                "description": adapted["description"],
                "estimated_seconds": adapted["estimated_seconds"],
                "support_note": adapted["support_note"],
                "adapted_from": current_step.get("description"),
                "adapted_for": feedback,
            }
            steps[current_index] = replacement
            session["micro_steps"] = steps
            session["adaptation_log"].append({
                "feedback": feedback,
                "old_step": current_step,
                "new_step": replacement,
                "at": _now_iso(),
            })
            session["nudge"] = adapted.get("encouragement")

    if int(session.get("current_step_index", 0)) >= len(session.get("micro_steps", [])):
        session["status"] = "complete"
        session["finished_at"] = _now_iso()
        session["nudge"] = "You made contact with the task. That counts."
        session["session_summary"] = _completion_summary(session)

    session["updated_at"] = _now_iso()
    sessions[session_id] = session
    _save_sessions(sessions)
    return _session_response(session)


@app.post("/api/support", response_model=SupportResponse)
async def support(request: SupportRequest):
    """Return a contextual support message, reminder, or nudge."""
    task = request.task.strip()
    if has_ai_provider():
        try:
            return create_support_message(
                task=task,
                current_step=request.current_step,
                moment=request.moment,
                user_state=request.user_state,
            )
        except Exception as e:
            print(f"AI support message failed, using fallback: {e}")

    return create_support_message_fallback(
        task=task,
        current_step=request.current_step,
        moment=request.moment,
        user_state=request.user_state,
    )


@app.post("/api/coworking/chat", response_model=CoworkingChatResponse)
async def coworking_chat(request: CoworkingChatRequest):
    """Return simulated coworking room messages."""
    mode = request.mode.strip().lower()
    if mode not in {"periodic", "reply"}:
        raise HTTPException(status_code=422, detail="Invalid coworking chat mode")

    task = request.task.strip()
    coworkers = [coworker.model_dump() for coworker in request.coworkers[:5]]
    recent_messages = [message.model_dump() for message in request.recent_messages[-30:]]

    if has_ai_provider():
        try:
            return create_coworking_messages(
                task=task,
                mode=mode,
                coworkers=coworkers,
                recent_messages=recent_messages,
                user_message=request.user_message,
                session_minutes=request.session_minutes,
            )
        except Exception as e:
            print(f"AI coworking message failed, using fallback: {e}")

    return create_coworking_messages_fallback(
        task=task,
        mode=mode,
        coworkers=coworkers,
        recent_messages=recent_messages,
        user_message=request.user_message,
        session_minutes=request.session_minutes,
    )


@app.get("/")
async def root():
    return {
        "name": "Unclump API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoint": "POST /api/breakdown",
        "session_endpoint": "POST /api/session/start",
        "support_endpoint": "POST /api/support",
        "coworking_endpoint": "POST /api/coworking/chat",
        "ai_provider_configured": has_ai_provider(),
    }


# ── Simple email waitlist (JSON file for persistence) ──

WAITLIST_FILE = _Path("waitlist.json")
SESSIONS_FILE = _Path("sessions.json")


def _load_waitlist() -> list[str]:
    if WAITLIST_FILE.exists():
        return _json.loads(WAITLIST_FILE.read_text())
    return []


def _save_waitlist(emails: list[str]) -> None:
    WAITLIST_FILE.write_text(_json.dumps(emails))


def _load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        return _json.loads(SESSIONS_FILE.read_text())
    return {}


def _save_sessions(sessions: dict) -> None:
    SESSIONS_FILE.write_text(_json.dumps(sessions, indent=2))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_response(session: dict) -> dict:
    steps = session.get("micro_steps", [])
    current_index = int(session.get("current_step_index", 0))
    current_step = steps[current_index] if current_index < len(steps) else None
    return {
        "session_id": session["session_id"],
        "status": session.get("status", "active"),
        "original_task": session["original_task"],
        "block_type": session["block_type"],
        "block_label": session["block_label"],
        "block_reason": session["block_reason"],
        "entry_hook": session["entry_hook"],
        "reflection_prompt": session.get("reflection_prompt", ""),
        "current_step_index": current_index,
        "completed_steps": int(session.get("completed_steps", 0)),
        "total_steps": len(steps),
        "current_step": current_step,
        "micro_steps": steps,
        "feedback_options": ["done", "too_hard", "need_smaller", "distracted", "not_right", "skip"],
        "nudge": session.get("nudge"),
        "session_summary": session.get("session_summary"),
    }


def _completion_summary(session: dict) -> str:
    completed = int(session.get("completed_steps", 0))
    total = len(session.get("micro_steps", []))
    block_label = session.get("block_label", "the stuck point")
    return (
        f"You completed {completed} of {total} steps and learned that "
        f"{block_label.lower()} may be part of this task's friction."
    )


class WaitlistRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)


@app.post("/api/waitlist")
async def waitlist_join(request: WaitlistRequest):
    email = request.email.strip().lower()
    emails = _load_waitlist()
    if email not in emails:
        emails.append(email)
        _save_waitlist(emails)
        print(f"[WAITLIST] New signup: {email} (total: {len(emails)})")
    return {"status": "ok", "message": "You're on the list!", "total": len(emails)}


@app.get("/api/waitlist")
async def waitlist_view():
    """View signups — for you to check who's joined."""
    emails = _load_waitlist()
    return {"total": len(emails), "emails": emails}
