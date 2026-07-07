"""Tests for the AI task breakdown engine."""

import json
import pytest
from unittest.mock import MagicMock, patch

from engine import (
    breakdown_task,
    breakdown_task_fallback,
    _parse_response,
    TaskBreakdown,
    MicroStep,
    SYSTEM_PROMPT,
)


class TestParseResponse:
    """Unit tests for _parse_response — no API calls."""

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
