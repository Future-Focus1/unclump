"""Unstuck — AI Task Breakdown Service.

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


def breakdown_task(
    task: str,
    client: OpenAI | None = None,
    model: str = "gpt-4o-mini",
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

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it via: export OPENAI_API_KEY=sk-..."
            )
        client = OpenAI(api_key=api_key)

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
