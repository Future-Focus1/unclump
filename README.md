# Unstuck

**The app that helps you start.**

Unstuck is an ADHD-friendly productivity app that solves the "execution chasm" — the gap between knowing what to do and actually doing it.

## How It Works

1. **Tell us what you're stuck on** — type it, say it, or WhatsApp it
2. **AI breaks it into micro-steps** — step 1 is always so small you can't fail
3. **You start** — gentle accountability, body doubling, no guilt

## Architecture

```
backend/
├── engine.py       # AI task breakdown engine (OpenAI + fallback)
├── main.py         # FastAPI server
├── test_engine.py  # Engine unit tests (21 tests)
└── test_api.py     # API integration tests (7 tests)
```

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000/docs
```

## API

```
POST /api/breakdown
{
  "task": "clean my room"
}
→ {
  "original_task": "clean my room",
  "entry_hook": "Let's go!",
  "total_estimated_minutes": 2.5,
  "micro_steps": [
    {"step": 1, "description": "Pick up one item", "estimated_seconds": 30, "is_entry_point": true},
    ...
  ]
}
```

## Status

- [x] AI task breakdown engine (with tests)
- [x] FastAPI server with fallback
- [x] Landing page
- [ ] Reddit validation
- [ ] Flutter mobile app
- [ ] WhatsApp bot
- [ ] Body doubling
