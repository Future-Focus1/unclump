"""Tests for the AI task breakdown engine."""

import json
import pytest
from unittest.mock import MagicMock, patch

from engine import (
    breakdown_task,
    breakdown_task_fallback,
    create_adaptive_step_fallback,
    create_coworking_messages_fallback,
    create_support_message_fallback,
    create_unclump_session_plan_fallback,
    _parse_coworking_messages,
    _parse_response,
    _parse_support_message,
    _parse_session_response,
    _provider_request_options,
    TaskBreakdown,
    MicroStep,
    SYSTEM_PROMPT,
)


class TestParseResponse:
    """Unit tests for _parse_response — no API calls."""

    def test_provider_options_disable_deepseek_thinking(self):
        assert _provider_request_options("deepseek") == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }
        assert _provider_request_options("openai") == {}

    def test_parses_valid_json(self):
        raw = json.dumps({
            "entry_hook": "You've got this.",
            "micro_steps": [
                {"description": "Open the laptop lid.", "estimated_seconds": 10},
                {"description": "Press the power button.", "estimated_seconds": 5},
            ],
        })
        result = _parse_response("write an email", raw)
        assert len(result.micro_steps) == 2
        assert result.micro_steps[0].description == "Open the laptop lid."
        assert result.micro_steps[0].is_entry_point is True
        assert result.micro_steps[1].is_entry_point is False
        assert result.entry_hook == "You've got this."

    def test_first_step_is_entry_point(self):
        raw = json.dumps({
            "entry_hook": "Go.",
            "micro_steps": [{"description": "Do it.", "estimated_seconds": 10}],
        })
        result = _parse_response("anything", raw)
        assert result.micro_steps[0].is_entry_point is True

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"entry_hook": "Hi", "micro_steps": [{"description": "X", "estimated_seconds": 10}]}\n```'
        result = _parse_response("task", raw)
        assert len(result.micro_steps) == 1

    def test_caps_at_6_steps(self):
        steps = [{"description": f"Step {i}", "estimated_seconds": 10} for i in range(10)]
        raw = json.dumps({"entry_hook": "Go", "micro_steps": steps})
        result = _parse_response("task", raw)
        assert len(result.micro_steps) == 6

    def test_clamps_estimated_seconds(self):
        raw = json.dumps({
            "entry_hook": "Go",
            "micro_steps": [
                {"description": "Too fast", "estimated_seconds": 2},
                {"description": "Too slow", "estimated_seconds": 300},
                {"description": "Just right", "estimated_seconds": 60},
            ],
        })
        result = _parse_response("task", raw)
        assert result.micro_steps[0].estimated_seconds == 10  # clamped up
        assert result.micro_steps[1].estimated_seconds == 120  # clamped down
        assert result.micro_steps[2].estimated_seconds == 60   # unchanged

    def test_total_estimated_seconds(self):
        raw = json.dumps({
            "entry_hook": "Go",
            "micro_steps": [
                {"description": "A", "estimated_seconds": 30},
                {"description": "B", "estimated_seconds": 45},
            ],
        })
        result = _parse_response("task", raw)
        assert result.total_estimated_seconds == 75

    def test_raises_on_empty_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_response("task", "not json at all {{{")

    def test_raises_on_missing_micro_steps(self):
        with pytest.raises(ValueError, match="Missing or invalid micro_steps"):
            _parse_response("task", '{"entry_hook": "Hi"}')

    def test_raises_on_empty_micro_steps(self):
        with pytest.raises(ValueError, match="zero micro-steps"):
            _parse_response("task", '{"entry_hook": "Hi", "micro_steps": []}')

    def test_raises_on_empty_step_description(self):
        raw = json.dumps({
            "entry_hook": "Go",
            "micro_steps": [{"description": "", "estimated_seconds": 10}],
        })
        with pytest.raises(ValueError, match="empty description"):
            _parse_response("task", raw)

    def test_to_dict_format(self):
        raw = json.dumps({
            "entry_hook": "Let's begin.",
            "micro_steps": [
                {"description": "Stand up.", "estimated_seconds": 5},
                {"description": "Walk to desk.", "estimated_seconds": 30},
            ],
        })
        result = _parse_response("get to work", raw)
        d = result.to_dict()
        assert d["original_task"] == "get to work"
        assert d["entry_hook"] == "Let's begin."
        assert d["total_estimated_minutes"] == pytest.approx(0.6, abs=0.1)
        assert len(d["micro_steps"]) == 2
        assert d["micro_steps"][0]["is_entry_point"] is True
        assert d["micro_steps"][1]["is_entry_point"] is False


class TestBreakdownTask:
    """Integration-style tests with mocked OpenAI client."""

    def test_breakdown_with_mock_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "entry_hook": "Let's go.",
                "micro_steps": [
                    {"description": "Open your eyes.", "estimated_seconds": 5},
                ],
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = breakdown_task("wake up", client=mock_client)
        assert result.original_task == "wake up"
        assert len(result.micro_steps) == 1
        assert result.micro_steps[0].description == "Open your eyes."

    def test_passes_system_prompt(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "entry_hook": "Go.",
                "micro_steps": [{"description": "X", "estimated_seconds": 10}],
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        breakdown_task("test", client=mock_client)

        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert "ADHD executive function coach" in system_msg
        assert "Step 1 must be a physical action" in system_msg

    def test_raises_on_empty_task(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            breakdown_task("", client=MagicMock())

        with pytest.raises(ValueError, match="cannot be empty"):
            breakdown_task("   ", client=MagicMock())

    def test_raises_on_api_failure(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")

        with pytest.raises(RuntimeError, match="API call failed"):
            breakdown_task("do something", client=mock_client)


class TestFallbackBreakdown:
    """Unit tests for the AI-free fallback breakdown."""

    def test_matches_keyword_clean(self):
        result = breakdown_task_fallback("I need to clean my room")
        assert "Pick up" in result.micro_steps[0].description
        assert result.micro_steps[0].is_entry_point is True

    def test_matches_keyword_email(self):
        result = breakdown_task_fallback("send an email to my boss")
        assert "email" in result.micro_steps[0].description.lower()

    def test_matches_keyword_workout(self):
        result = breakdown_task_fallback("do a workout")
        assert "Stand up" in result.micro_steps[0].description

    def test_generic_fallback_for_unknown(self):
        result = breakdown_task_fallback("something completely unknown xyz")
        assert len(result.micro_steps) > 0
        assert result.micro_steps[0].is_entry_point is True
        assert "something completely unknown xyz" in result.micro_steps[0].description
        descriptions = " ".join(step.description for step in result.micro_steps)
        assert "Notice: you've started" not in descriptions
        assert "Do the next visible action" not in descriptions

    def test_all_steps_have_valid_estimates(self):
        result = breakdown_task_fallback("clean")
        for step in result.micro_steps:
            assert 5 <= step.estimated_seconds <= 120

    def test_total_seconds_calculated(self):
        result = breakdown_task_fallback("clean")
        assert result.total_estimated_seconds > 0
        assert result.total_estimated_seconds == sum(
            s.estimated_seconds for s in result.micro_steps
        )


class TestUnclumpSessionPlan:
    """Tests for adaptive session planning."""

    def test_parse_session_response(self):
        raw = json.dumps({
            "block_type": "unclear_next_step",
            "block_label": "Unclear next step",
            "block_reason": "The next action is fuzzy.",
            "confidence": 0.8,
            "entry_hook": "Start with one visible move.",
            "reflection_prompt": "What helped?",
            "micro_steps": [
                {
                    "description": "Open a blank note.",
                    "estimated_seconds": 5,
                    "support_note": "This makes the task visible.",
                }
            ],
        })
        result = _parse_session_response("start project", raw)
        assert result.block_type == "unclear_next_step"
        assert result.confidence == 0.8
        assert result.micro_steps[0].estimated_seconds == 15
        assert result.micro_steps[0].support_note == "This makes the task visible."
        assert result.block_label == "Let's break it down - we can handle this in one go if we break it into smaller chunks."

    def test_parse_session_response_accepts_large_task_route(self):
        raw = json.dumps({
            "task_size": "large",
            "planning_mode": "multi_session_project",
            "block_type": "overwhelm",
            "block_reason": "This needs a first checkpoint.",
            "confidence": 0.8,
            "entry_hook": "Start with the first checkpoint.",
            "session_goal": "Create the first business note.",
            "stopping_point": "Stop when the next action is named.",
            "progress_notes": ["Business idea named."],
            "next_session_prompt": "Continue from the named action.",
            "reflection_prompt": "What moved?",
            "micro_steps": [
                {
                    "description": "Write the business idea in one sentence.",
                    "estimated_seconds": 300,
                    "support_note": "This makes the project visible.",
                }
            ],
        })
        result = _parse_session_response("start a company", raw)
        assert result.task_size == "large"
        assert result.planning_mode == "multi_session_project"
        assert result.block_label == "Overwhelming - this may feel too large to do in one go, let's do it over multiple sessions together."
        assert result.stopping_point == "Stop when the next action is named."

    def test_fallback_detects_low_energy(self):
        result = create_unclump_session_plan_fallback(
            "I am too exhausted to start laundry",
            context={"energy_level": "low"},
        )
        assert result.block_type == "low_energy"
        assert result.micro_steps[0].is_entry_point is True
        assert "laundry" in " ".join(step.description.lower() for step in result.micro_steps)

    def test_fallback_classifies_leave_house_as_piece_of_cake(self):
        result = create_unclump_session_plan_fallback("leave the house")
        assert result.task_size == "small"
        assert result.planning_mode == "single_burst"
        assert result.block_label == "Piece of Cake - You've got this one."
        assert 1 <= len(result.micro_steps) <= 3

    def test_fallback_classifies_start_company_as_multi_session(self):
        result = create_unclump_session_plan_fallback("start a company")
        assert result.task_size == "large"
        assert result.planning_mode == "multi_session_project"
        assert result.stopping_point
        assert any("business idea" in step.description.lower() for step in result.micro_steps)

    def test_session_fallback_uses_task_context_for_unknown(self):
        result = create_unclump_session_plan_fallback(
            "sort the confusing visa documents",
            context={"recent_friction": "too much"},
        )
        descriptions = " ".join(step.description for step in result.micro_steps)
        assert "sort the confusing visa documents" in descriptions
        assert "Do the next visible action" not in descriptions

    def test_adaptive_fallback_makes_step_smaller(self):
        current_step = {
            "step": 1,
            "description": "Write the whole email.",
            "estimated_seconds": 120,
        }
        result = create_adaptive_step_fallback(
            "reply to an email",
            current_step,
            "too_hard",
            "overwhelm",
        )
        assert result["estimated_seconds"] <= 15
        assert result["description"] == "Open the message and read the sender's name."

    def test_parse_support_message(self):
        raw = json.dumps({
            "message": "Small is enough.",
            "suggested_action": "Open the draft.",
            "reminder_after_minutes": 5,
            "tone": "gentle",
        })
        result = _parse_support_message(raw)
        assert result["message"] == "Small is enough."
        assert result["suggested_action"] == "Open the draft."
        assert result["reminder_after_minutes"] == 5

    def test_support_fallback_is_contextual(self):
        result = create_support_message_fallback(
            "reply to an email",
            moment="nudge",
            user_state="stuck",
        )
        assert "sender" in result["suggested_action"]
        assert "reply" in result["message"]

        clean_result = create_support_message_fallback(
            "clean the kitchen counter",
            moment="nudge",
            user_state="overwhelmed",
        )
        assert "visible patch" in clean_result["message"]

    def test_parse_coworking_messages_limits_sender_and_length(self):
        raw = json.dumps({
            "messages": [
                {
                    "sender": "NotInRoom",
                    "text": " ".join(["word"] * 40),
                }
            ]
        })
        result = _parse_coworking_messages(raw, [{"name": "Maya"}])
        assert result["messages"][0]["sender"] == "Maya"
        assert len(result["messages"][0]["text"].split()) <= 30

    def test_coworking_fallback_reply_is_short_and_supportive(self):
        result = create_coworking_messages_fallback(
            task="reply to an email",
            mode="reply",
            coworkers=[
                {"name": "Maya", "task": "sorting receipts", "quirk": "uses_2"},
                {"name": "Sam", "task": "opening a draft", "quirk": "lol"},
            ],
            user_message="I'm stuck, what should I do?",
        )
        assert 1 <= len(result["messages"]) <= 2
        assert all(len(message["text"].split()) <= 30 for message in result["messages"])
        assert any(message["sender"] in {"Maya", "Sam"} for message in result["messages"])
