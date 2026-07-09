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
    prior_progress: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional saved progress from a previous related session.",
    )
    previous_stopping_point: str | None = Field(
        default=None,
        max_length=500,
        description="Optional stopping point from a previous related session.",
    )


class SessionFeedbackRequest(BaseModel):
    feedback: str = Field(
        ...,
        description="done, need_smaller, distracted, or skip",
    )


class SessionResponse(BaseModel):
    session_id: str
    status: str
    original_task: str
    block_type: str
    block_label: str
    block_reason: str
    task_size: str = "medium"
    planning_mode: str = "one_session_steps"
    task_intent: str = "immediate_action"
    task_domain: str = "general"
    session_goal: str = ""
    stopping_point: str | None = None
    safety_note: str | None = None
    progress_notes: list[str] = Field(default_factory=list)
    next_session_prompt: str | None = None
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
    adhd_trait: str | None = Field(default=None, max_length=140)


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
        "prior_progress": request.prior_progress,
        "previous_stopping_point": request.previous_stopping_point,
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
        "progress_notes": plan.progress_notes,
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
        if current_step:
            session.setdefault("progress_notes", []).append(
                f"Completed: {current_step.get('description', 'step')}"
            )
        session["nudge"] = "Good. That counted. Next move."
    elif feedback == "skip":
        session["current_step_index"] = current_index + 1
        session["nudge"] = "Skipped. Fine. Keep momentum with the next useful move."
    else:
        if not current_step:
            session["status"] = "complete"
            session["session_summary"] = _completion_summary(session)
        else:
            if feedback == "need_smaller":
                replan_context = {
                    "preferred_tone": "direct_motivational",
                    "recent_friction": (
                        "The user hit Make smaller. Replan only the remaining work. "
                        f"Current step was too big: {current_step.get('description', '')}. "
                        f"Completed so far: {session.get('progress_notes', [])[-6:]}. "
                        f"Original stopping point: {session.get('stopping_point') or 'none'}."
                    ),
                    "prior_progress": "; ".join(session.get("progress_notes", [])[-8:]),
                    "previous_stopping_point": session.get("stopping_point"),
                }
                if has_ai_provider():
                    try:
                        replanned = create_unclump_session_plan(
                            session["original_task"],
                            context=replan_context,
                        )
                    except Exception as e:
                        print(f"AI replan failed, using fallback: {e}")
                        replanned = create_unclump_session_plan_fallback(
                            session["original_task"],
                            context=replan_context,
                        )
                else:
                    replanned = create_unclump_session_plan_fallback(
                        session["original_task"],
                        context=replan_context,
                    )

                replacement_steps = replanned.to_dict()["micro_steps"]
                if replacement_steps:
                    smaller = create_adaptive_step_fallback(
                        session["original_task"],
                        current_step,
                        feedback,
                        session["block_type"],
                    )
                    replacement_steps[0] = {
                        **replacement_steps[0],
                        "description": smaller["description"],
                        "estimated_seconds": smaller["estimated_seconds"],
                        "support_note": smaller["support_note"],
                        "adapted_from": current_step.get("description"),
                        "adapted_for": feedback,
                    }
                kept_steps = steps[:current_index]
                renumbered = []
                for offset, step in enumerate(replacement_steps[:6], start=current_index + 1):
                    renumbered.append({**step, "step": offset})
                session["micro_steps"] = kept_steps + renumbered
                session["task_size"] = replanned.task_size
                session["planning_mode"] = replanned.planning_mode
                session["task_intent"] = replanned.task_intent
                session["task_domain"] = replanned.task_domain
                session["block_label"] = replanned.block_label
                session["block_reason"] = replanned.block_reason
                session["session_goal"] = replanned.session_goal
                session["stopping_point"] = replanned.stopping_point
                session["safety_note"] = replanned.safety_note
                session["next_session_prompt"] = replanned.next_session_prompt
                session.setdefault("progress_notes", []).append(
                    "Replanned the remaining route into smaller moves."
                )
                session["adaptation_log"].append({
                    "feedback": feedback,
                    "old_step": current_step,
                    "new_step": replacement_steps[0] if replacement_steps else None,
                    "at": _now_iso(),
                })
                session["nudge"] = "Smaller route loaded. Same mission."
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
        session["nudge"] = "Done for this round. Progress recorded."
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
    """Return virtual coworking room messages."""
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
        "task_size": session.get("task_size", "medium"),
        "planning_mode": session.get("planning_mode", "one_session_steps"),
        "task_intent": session.get("task_intent", "immediate_action"),
        "task_domain": session.get("task_domain", "general"),
        "session_goal": session.get("session_goal", ""),
        "stopping_point": session.get("stopping_point"),
        "safety_note": session.get("safety_note"),
        "progress_notes": session.get("progress_notes", []),
        "next_session_prompt": session.get("next_session_prompt"),
        "entry_hook": session["entry_hook"],
        "reflection_prompt": session.get("reflection_prompt", ""),
        "current_step_index": current_index,
        "completed_steps": int(session.get("completed_steps", 0)),
        "total_steps": len(steps),
        "current_step": current_step,
        "micro_steps": steps,
        "feedback_options": ["done", "need_smaller", "distracted", "skip"],
        "nudge": session.get("nudge"),
        "session_summary": session.get("session_summary"),
    }


def _completion_summary(session: dict) -> str:
    completed = int(session.get("completed_steps", 0))
    total = len(session.get("micro_steps", []))
    if session.get("planning_mode") == "multi_session_project":
        stopping_point = session.get("stopping_point") or "the next action is named"
        return (
            f"Session checkpoint reached: {completed} of {total} steps done. "
            f"Progress is recorded; next time, pick up from: {stopping_point}"
        )
    return f"You completed {completed} of {total} steps. Good. That task is no longer frozen."


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
