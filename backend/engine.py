"""Unclump — AI Task Breakdown Service.

Core engine that turns a user's vague task into ADHD-friendly micro-steps.
The key insight: Step 1 must be so trivially small it bypasses task paralysis.
"""

from dataclasses import dataclass, field
import json
import os
from openai import OpenAI


@dataclass
class MicroStep:
    """A single action so small you can't fail to do it."""
    step_number: int
    description: str
    estimated_seconds: int  # max 120 — keep it tiny
    is_entry_point: bool = False  # the first physical action


@dataclass
class TaskBreakdown:
    """A task broken into micro-steps optimized for ADHD brains."""
    original_task: str
    micro_steps: list[MicroStep] = field(default_factory=list)
    total_estimated_seconds: int = 0
    entry_hook: str = ""  # one-sentence nudge to start

    def to_dict(self) -> dict:
        return {
            "original_task": self.original_task,
            "entry_hook": self.entry_hook,
            "total_estimated_minutes": round(self.total_estimated_seconds / 60, 1),
            "micro_steps": [
                {
                    "step": s.step_number,
                    "description": s.description,
                    "estimated_seconds": s.estimated_seconds,
                    "is_entry_point": s.is_entry_point,
                }
                for s in self.micro_steps
            ],
        }


@dataclass
class SessionStep:
    """A step with enough context for the app to coach around it."""
    step_number: int
    description: str
    estimated_seconds: int
    support_note: str = ""
    is_entry_point: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "description": self.description,
            "estimated_seconds": self.estimated_seconds,
            "support_note": self.support_note,
            "is_entry_point": self.is_entry_point,
        }


@dataclass
class UnclumpSessionPlan:
    """A diagnosed task-rescue plan for an interactive Unclump session."""
    original_task: str
    block_type: str
    block_label: str
    block_reason: str
    confidence: float
    entry_hook: str
    micro_steps: list[SessionStep] = field(default_factory=list)
    reflection_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "original_task": self.original_task,
            "block_type": self.block_type,
            "block_label": self.block_label,
            "block_reason": self.block_reason,
            "confidence": self.confidence,
            "entry_hook": self.entry_hook,
            "reflection_prompt": self.reflection_prompt,
            "micro_steps": [step.to_dict() for step in self.micro_steps],
        }


VALID_BLOCK_TYPES = {
    "overwhelm": "Overwhelm",
    "unclear_next_step": "Unclear next step",
    "shame": "Shame spiral",
    "low_energy": "Low energy",
    "avoidance": "Avoidance",
    "sensory": "Sensory friction",
    "emotional": "Emotional load",
}


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
OPENAI_MODEL = "gpt-4o-mini"


def has_ai_provider() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _get_ai_client_and_model(
    client: OpenAI | None = None,
    model: str | None = None,
) -> tuple[OpenAI, str, str]:
    """Return the configured AI client, model, and provider name."""
    if client is not None:
        return client, model or OPENAI_MODEL, "custom"

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return (
            OpenAI(
                api_key=deepseek_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            ),
            model or os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
            "deepseek",
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key), model or OPENAI_MODEL, "openai"

    raise RuntimeError(
        "No AI provider configured. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
    )


SYSTEM_PROMPT = """You are an ADHD executive function coach. Your job is to break down a task 
into the smallest possible micro-steps so someone with task paralysis can actually start.

CRITICAL RULES:
1. Step 1 must be a physical action so trivial it requires ZERO motivation. 
   NOT "plan your approach" or "think about" — those are cognitive, not physical.
   GOOD: "Open the laptop lid." "Pick up the pen." "Stand up from the chair."
2. Every step must be a concrete, observable action. No abstract steps.
3. Maximum 6 steps. Each step should take under 2 minutes.
4. Never use the words "just" or "simply" — they shame the user.
5. End with a clear completion action that signals "done."

Respond with ONLY valid JSON in this exact format:
{
  "entry_hook": "One gentle sentence inviting them to start. Warm, not commanding.",
  "micro_steps": [
    {"description": "Physical action (under 2 min)", "estimated_seconds": 30},
    ...
  ]
}

estimated_seconds must be between 10 and 120. Be realistic.
"""


SESSION_SYSTEM_PROMPT = """You are Unclump, an ADHD-friendly execution rescue coach.

The user is stuck. Your job is not to plan their life. Your job is to diagnose the likely
kind of stuckness and create the first few actions that can get them moving right now.

CRITICAL RULES:
1. Step 1 must be a tiny physical action requiring almost no motivation.
2. Never shame, scold, moralize, or use the words "just" or "simply".
3. Keep steps concrete and observable. Avoid abstract planning words.
4. Maximum 6 steps. Each step must be 10-120 seconds.
5. Include a short support_note for each step that explains why this tiny action helps.
6. This is productivity support, not therapy or medical advice.

Classify block_type as one of:
overwhelm, unclear_next_step, shame, low_energy, avoidance, sensory, emotional

Respond with ONLY valid JSON in this exact shape:
{
  "block_type": "overwhelm",
  "block_label": "Overwhelm",
  "block_reason": "One short non-clinical reason this task may feel stuck.",
  "confidence": 0.7,
  "entry_hook": "One warm sentence inviting the first action.",
  "reflection_prompt": "One gentle question to ask after the session.",
  "micro_steps": [
    {
      "description": "Tiny physical action.",
      "estimated_seconds": 30,
      "support_note": "One sentence explaining why this is safe to start."
    }
  ]
}
"""


ADAPTATION_SYSTEM_PROMPT = """You are Unclump, an ADHD-friendly execution rescue coach.

The user pushed back on the current step. Replace it with one better step.
Make the replacement smaller, more concrete, and emotionally safer.

Rules:
1. Never shame, scold, moralize, or use the words "just" or "simply".
2. Return one physical action only.
3. estimated_seconds must be 10-120.
4. If the user says distracted, give a reset action.
5. If the user says too_hard or need_smaller, shrink the action dramatically.
6. If the user says not_right, change direction without arguing.

Respond with ONLY valid JSON:
{
  "description": "Replacement physical action.",
  "estimated_seconds": 20,
  "support_note": "Gentle reason this step may be easier.",
  "encouragement": "One short warm sentence."
}
"""


SUPPORT_SYSTEM_PROMPT = """You are Unclump, an ADHD-friendly execution support coach.

Write a short contextual support message for someone who is stuck on a task.
This is productivity support, not therapy or medical advice.

Rules:
1. Be warm, concrete, and brief.
2. Never shame, scold, moralize, or use the words "just" or "simply".
3. If giving a reminder, make it a gentle invitation, not a command.
4. If giving a nudge, include one tiny physical next action.
5. If the user is distressed, validate and reduce the demand.

Respond with ONLY valid JSON:
{
  "message": "Short support message.",
  "suggested_action": "One tiny physical action.",
  "reminder_after_minutes": 10,
  "tone": "gentle"
}
"""


def breakdown_task(
    task: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> TaskBreakdown:
    """Break a task into ADHD-friendly micro-steps using AI.

    Args:
        task: The raw task description from the user.
        client: OpenAI client. If None, creates one from env var.
        model: Model to use. Defaults to gpt-4o-mini for speed + cost.

    Returns:
        TaskBreakdown with micro-steps.

    Raises:
        ValueError: If the AI response can't be parsed or is empty.
        RuntimeError: If the API call fails.
    """
    if not task or not task.strip():
        raise ValueError("Task description cannot be empty")

    client, model, _provider = _get_ai_client_and_model(client, model)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Break down this task: {task}"},
            ],
            temperature=0.4,  # low temp for consistency
            max_tokens=500,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    raw = response.choices[0].message.content or ""
    return _parse_response(task, raw)


def _parse_response(task: str, raw: str) -> TaskBreakdown:
    """Parse the AI response into a TaskBreakdown, with validation."""
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI returned invalid JSON. Raw response: {raw[:200]}..."
        ) from e

    if "micro_steps" not in data or not isinstance(data["micro_steps"], list):
        raise ValueError(f"Missing or invalid micro_steps in response: {data}")

    if len(data["micro_steps"]) == 0:
        raise ValueError("AI returned zero micro-steps")

    if len(data["micro_steps"]) > 6:
        data["micro_steps"] = data["micro_steps"][:6]  # enforce cap

    steps = []
    total = 0
    for i, step_data in enumerate(data["micro_steps"], 1):
        desc = str(step_data.get("description", "")).strip()
        if not desc:
            raise ValueError(f"Step {i} has empty description")

        est = int(step_data.get("estimated_seconds", 60))
        est = max(10, min(120, est))  # clamp

        steps.append(MicroStep(
            step_number=i,
            description=desc,
            estimated_seconds=est,
            is_entry_point=(i == 1),
        ))
        total += est

    entry_hook = str(data.get("entry_hook", "Ready when you are."))

    return TaskBreakdown(
        original_task=task,
        micro_steps=steps,
        total_estimated_seconds=total,
        entry_hook=entry_hook,
    )


def create_unclump_session_plan(
    task: str,
    context: dict | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
) -> UnclumpSessionPlan:
    """Create a diagnosed, interactive task rescue plan."""
    if not task or not task.strip():
        raise ValueError("Task description cannot be empty")

    client, model, _provider = _get_ai_client_and_model(client, model)

    context = context or {}
    context_lines = [
        f"Task: {task}",
        f"Energy level: {context.get('energy_level', 'unknown')}",
        f"Preferred tone: {context.get('preferred_tone', 'gentle')}",
        f"Recent friction: {context.get('recent_friction', 'unknown')}",
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SESSION_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(context_lines)},
            ],
            temperature=0.35,
            max_tokens=800,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    raw = response.choices[0].message.content or ""
    return _parse_session_response(task, raw)


def _parse_session_response(task: str, raw: str) -> UnclumpSessionPlan:
    """Parse an AI session response into a validated session plan."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI returned invalid JSON. Raw response: {raw[:200]}..."
        ) from e

    if "micro_steps" not in data or not isinstance(data["micro_steps"], list):
        raise ValueError(f"Missing or invalid micro_steps in response: {data}")
    if not data["micro_steps"]:
        raise ValueError("AI returned zero micro-steps")

    block_type = str(data.get("block_type", "overwhelm")).strip().lower()
    if block_type not in VALID_BLOCK_TYPES:
        block_type = "overwhelm"

    steps = []
    for i, step_data in enumerate(data["micro_steps"][:6], 1):
        desc = str(step_data.get("description", "")).strip()
        if not desc:
            raise ValueError(f"Step {i} has empty description")
        est = int(step_data.get("estimated_seconds", 60))
        est = max(10, min(120, est))
        support_note = str(step_data.get("support_note", "")).strip()
        steps.append(SessionStep(
            step_number=i,
            description=desc,
            estimated_seconds=est,
            support_note=support_note,
            is_entry_point=(i == 1),
        ))

    confidence = float(data.get("confidence", 0.6))
    confidence = max(0.0, min(1.0, confidence))

    return UnclumpSessionPlan(
        original_task=task,
        block_type=block_type,
        block_label=str(data.get("block_label") or VALID_BLOCK_TYPES[block_type]),
        block_reason=str(data.get("block_reason") or "This task may need a smaller entry point."),
        confidence=confidence,
        entry_hook=str(data.get("entry_hook") or "Let's make the first move tiny."),
        reflection_prompt=str(data.get("reflection_prompt") or "What made starting a little easier?"),
        micro_steps=steps,
    )


def create_unclump_session_plan_fallback(
    task: str,
    context: dict | None = None,
) -> UnclumpSessionPlan:
    """Heuristic session plan when AI is unavailable."""
    context = context or {}
    block_type = _guess_block_type(task, context)
    breakdown = breakdown_task_fallback(task)
    task_lower = task.lower()

    support_note = {
        "overwhelm": "This lowers the size of the task until your brain has a doorway in.",
        "unclear_next_step": "This turns the fog into one visible action.",
        "shame": "This starts without asking you to fix everything at once.",
        "low_energy": "This uses a very low-energy movement to build momentum.",
        "avoidance": "This makes contact with the task without demanding a full commitment.",
        "sensory": "This reduces friction before asking for focus.",
        "emotional": "This gives your body a small action while the feelings catch up.",
    }[block_type]

    if block_type == "low_energy":
        first_steps = [
            ("Put both feet on the floor.", 10),
            ("Take one slow breath.", 10),
            ("Open the place where this task lives.", 20),
        ]
    elif block_type == "unclear_next_step":
        first_steps = [
            ("Open a blank note or the relevant app.", 15),
            ("Write one messy sentence about what needs to happen.", 30),
            ("Circle or copy the first visible action word.", 20),
        ]
    elif block_type == "shame":
        first_steps = [
            ("Put one hand on the object or app connected to this task.", 10),
            ("Say: this is allowed to be messy.", 10),
            ("Do one visible action for ten seconds.", 10),
        ]
    elif "email" in task_lower or "message" in task_lower or "reply" in task_lower:
        first_steps = [
            ("Open the message without replying yet.", 15),
            ("Read only the first two lines.", 20),
            ("Type one rough sentence in the reply box.", 45),
        ]
    else:
        first_steps = [
            (breakdown.micro_steps[0].description, min(30, breakdown.micro_steps[0].estimated_seconds)),
            ("Pause and notice that the task has started.", 10),
            ("Do the next visible action for thirty seconds.", 30),
        ]

    remaining = [
        (step.description, step.estimated_seconds)
        for step in breakdown.micro_steps[1:]
    ]
    combined_steps = (first_steps + remaining)[:6]
    micro_steps = [
        SessionStep(
            step_number=i,
            description=desc,
            estimated_seconds=max(10, min(120, est)),
            support_note=support_note if i == 1 else "Keep the next move visible and small.",
            is_entry_point=(i == 1),
        )
        for i, (desc, est) in enumerate(combined_steps, 1)
    ]

    return UnclumpSessionPlan(
        original_task=task,
        block_type=block_type,
        block_label=VALID_BLOCK_TYPES[block_type],
        block_reason=_fallback_block_reason(block_type),
        confidence=0.55,
        entry_hook="No whole task right now. Only the next tiny movement.",
        reflection_prompt="Which step felt easiest to start, even a little?",
        micro_steps=micro_steps,
    )


def create_adaptive_step(
    task: str,
    current_step: dict,
    feedback: str,
    block_type: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict:
    """Create a replacement step after the user says the current one is stuck."""
    client, model, _provider = _get_ai_client_and_model(client, model)

    prompt = {
        "task": task,
        "block_type": block_type,
        "feedback": feedback,
        "current_step": current_step,
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ADAPTATION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.35,
            max_tokens=300,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    raw = response.choices[0].message.content or ""
    return _parse_adaptive_step(raw)


def _parse_adaptive_step(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI returned invalid JSON. Raw response: {raw[:200]}..."
        ) from e

    desc = str(data.get("description", "")).strip()
    if not desc:
        raise ValueError("Adaptive step has empty description")
    est = int(data.get("estimated_seconds", 20))
    return {
        "description": desc,
        "estimated_seconds": max(10, min(120, est)),
        "support_note": str(data.get("support_note") or "This replacement keeps the action small."),
        "encouragement": str(data.get("encouragement") or "Good data. We can make it easier."),
    }


def create_adaptive_step_fallback(
    task: str,
    current_step: dict,
    feedback: str,
    block_type: str,
) -> dict:
    """Heuristic replacement step for adaptive sessions."""
    task_lower = task.lower()
    feedback = feedback.lower()

    if feedback in {"too_hard", "need_smaller"}:
        if "email" in task_lower or "message" in task_lower or "reply" in task_lower:
            description = "Open the message and read the sender's name."
        elif "clean" in task_lower or "tidy" in task_lower or "room" in task_lower:
            description = "Pick up one item you can see."
        elif "tax" in task_lower or "form" in task_lower or "admin" in task_lower:
            description = "Open the folder, app, or page for this task."
        elif "call" in task_lower or "phone" in task_lower:
            description = "Open the contact or number without calling yet."
        elif "write" in task_lower or "essay" in task_lower or "report" in task_lower:
            description = "Open a blank note and type one rough word."
        else:
            description = "Name the next visible action out loud."
        support_note = "This lowers the demand to one concrete point of contact."
        estimated_seconds = 10
    elif feedback == "distracted":
        description = "Look away from the screen and take one slow breath."
        support_note = "This resets attention before choosing the next tiny action."
        estimated_seconds = 10
    elif feedback == "not_right":
        if "email" in task_lower or "message" in task_lower or "reply" in task_lower:
            description = "Open the message and read the sender's name."
        else:
            description = "Name one object, app, or page connected to this task."
        support_note = "Changing direction is allowed; the goal is to find a truer entry point."
        estimated_seconds = 15
    else:
        description = "Do the smallest visible part of the current step for ten seconds."
        support_note = "Ten seconds is enough to restart without turning this into a big demand."
        estimated_seconds = 10

    if block_type == "shame" and feedback in {"too_hard", "need_smaller"}:
        description = "Say: this can be messy, then touch the task for ten seconds."
        support_note = "The action starts with permission instead of pressure."

    return {
        "description": description,
        "estimated_seconds": estimated_seconds,
        "support_note": support_note,
        "encouragement": "Good catch. The app should adapt to you, not the other way around.",
    }


def create_support_message(
    task: str,
    current_step: dict | None = None,
    moment: str = "nudge",
    user_state: str | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict:
    """Create a contextual support, reminder, or nudge message."""
    client, model, _provider = _get_ai_client_and_model(client, model)
    prompt = {
        "task": task,
        "current_step": current_step or {},
        "moment": moment,
        "user_state": user_state or "",
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.45,
            max_tokens=220,
        )
    except Exception as e:
        raise RuntimeError(f"AI support message failed: {e}") from e

    raw = response.choices[0].message.content or ""
    return _parse_support_message(raw)


def _parse_support_message(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI returned invalid JSON. Raw response: {raw[:200]}..."
        ) from e

    message = str(data.get("message", "")).strip()
    suggested_action = str(data.get("suggested_action", "")).strip()
    if not message:
        raise ValueError("Support message is empty")
    if not suggested_action:
        suggested_action = "Take one breath and choose the smallest visible move."
    reminder = int(data.get("reminder_after_minutes", 10))
    return {
        "message": message,
        "suggested_action": suggested_action,
        "reminder_after_minutes": max(1, min(120, reminder)),
        "tone": str(data.get("tone") or "gentle"),
    }


def create_support_message_fallback(
    task: str,
    current_step: dict | None = None,
    moment: str = "nudge",
    user_state: str | None = None,
) -> dict:
    """Heuristic support text when no AI provider is available."""
    current_step = current_step or {}
    task_lower = task.lower()
    user_state = (user_state or "").lower()

    if "distracted" in user_state or moment == "distracted":
        return {
            "message": "Attention wandered. That is information, not a verdict.",
            "suggested_action": "Look away from the screen and take one slow breath.",
            "reminder_after_minutes": 5,
            "tone": "gentle",
        }
    if "overwhelmed" in user_state or "too much" in user_state:
        return {
            "message": "The task is asking for too much at once. Shrink the doorway.",
            "suggested_action": "Do one visible part of it for ten seconds.",
            "reminder_after_minutes": 10,
            "tone": "gentle",
        }
    if "email" in task_lower or "message" in task_lower or "reply" in task_lower:
        action = "Open the message and read only the sender's name."
    elif "clean" in task_lower or "tidy" in task_lower or "room" in task_lower:
        action = "Pick up one item you can see."
    else:
        action = current_step.get("description") or "Name the next visible action."

    return {
        "message": "You do not need the whole task. You only need the next small contact point.",
        "suggested_action": action,
        "reminder_after_minutes": 10,
        "tone": "gentle",
    }


def _guess_block_type(task: str, context: dict | None = None) -> str:
    context = context or {}
    text = " ".join([
        task.lower(),
        str(context.get("recent_friction", "")).lower(),
        str(context.get("energy_level", "")).lower(),
    ])
    if any(word in text for word in ["tired", "exhausted", "drained", "low energy", "can't move"]):
        return "low_energy"
    if any(word in text for word in ["ashamed", "guilty", "behind", "late", "failed", "embarrassed"]):
        return "shame"
    if any(word in text for word in ["don't know", "where to start", "unclear", "confused"]):
        return "unclear_next_step"
    if any(word in text for word in ["noisy", "messy", "too bright", "sensory", "smell", "sound"]):
        return "sensory"
    if any(word in text for word in ["scared", "anxious", "upset", "angry", "sad"]):
        return "emotional"
    if any(word in text for word in ["avoid", "avoiding", "putting off", "procrastinating"]):
        return "avoidance"
    return "overwhelm"


def _fallback_block_reason(block_type: str) -> str:
    reasons = {
        "overwhelm": "The task may feel too large to enter all at once.",
        "unclear_next_step": "The next action may not be visible enough yet.",
        "shame": "Pressure and guilt can make starting feel unsafe.",
        "low_energy": "Your available energy may be lower than the task is asking for.",
        "avoidance": "The task may need a lower-commitment first contact.",
        "sensory": "The environment or setup may be adding extra friction.",
        "emotional": "The task may be carrying more feeling than its size suggests.",
    }
    return reasons.get(block_type, reasons["overwhelm"])


# ── Fallback: template-based breakdown when AI is unavailable ──

FALLBACK_TEMPLATES = {
    "clean": [
        ("Pick up one item from the floor.", 30),
        ("Put that item where it belongs.", 30),
        ("Pick up the next item you see.", 30),
        ("Put it away.", 30),
        ("Set a timer for 5 more minutes.", 10),
        ("Stop when the timer rings — you're done.", 10),
    ],
    "email": [
        ("Open your email app or tab.", 15),
        ("Click 'Compose' or 'New Email'.", 10),
        ("Type the recipient's name in the 'To' field.", 20),
        ("Write a one-sentence subject line.", 30),
        ("Write the body — even if it's just 'Here you go'.", 60),
        ("Click Send.", 10),
    ],
    "workout": [
        ("Stand up from your chair.", 10),
        ("Put on your workout clothes.", 60),
        ("Fill your water bottle.", 30),
        ("Walk to your workout spot.", 60),
        ("Do ONE exercise — just one set.", 60),
        ("Decide: keep going or you're done. Either is fine.", 10),
    ],
}


def breakdown_task_fallback(task: str) -> TaskBreakdown:
    """Template-based fallback when AI is unavailable."""
    task_lower = task.lower()
    best_match = None
    for keyword, steps in FALLBACK_TEMPLATES.items():
        if keyword in task_lower:
            best_match = steps
            break

    if best_match is None:
        # Generic fallback
        best_match = [
            ("Take one deep breath.", 10),
            ("Name the first physical action needed.", 15),
            ("Do that action now.", 30),
            ("Notice: you've started. That's the hard part.", 10),
            ("Decide your next move.", 30),
            ("Keep going or stop. You've already won.", 10),
        ]

    micro_steps = [
        MicroStep(
            step_number=i,
            description=desc,
            estimated_seconds=est,
            is_entry_point=(i == 1),
        )
        for i, (desc, est) in enumerate(best_match, 1)
    ]

    return TaskBreakdown(
        original_task=task,
        micro_steps=micro_steps,
        total_estimated_seconds=sum(s.estimated_seconds for s in micro_steps),
        entry_hook="Let's take this one tiny step at a time.",
    )
