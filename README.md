# Unclump

**Break through task paralysis.**

Unclump is an ADHD-friendly app for crossing the "execution chasm": the gap
between knowing what to do and actually starting.

## Live Endpoints

| Service | URL |
|---------|-----|
| Landing page | https://3bbfef3c3aac510e-94-13-62-56.serveousercontent.com |
| API health | https://39a564e2a78134da-94-13-62-56.serveousercontent.com/health |
| API docs | https://39a564e2a78134da-94-13-62-56.serveousercontent.com/docs |

## Quick API Test

```bash
curl -X POST https://39a564e2a78134da-94-13-62-56.serveousercontent.com/api/breakdown \
  -H "Content-Type: application/json" \
  -d '{"task":"reply to emails"}'
```

## Adaptive Session API

The product direction is the adaptive session loop: size the task, choose the
right route, assign realistic timers, then adapt when the user needs a smaller
route or has drifted. Small tasks get a short confidence route, medium tasks
are broken into one-session chunks, and large tasks become multi-session
projects with saved stopping points.

AI providers are server-side only. Configure `DEEPSEEK_API_KEY` on the backend
host to use DeepSeek V4 Pro (`DEEPSEEK_MODEL=deepseek-v4-pro`); if it is
missing, the backend can still fall back to `OPENAI_API_KEY` or deterministic
local heuristics.

The session planner now interprets the user's real-world goal before generating
steps. Outcome prompts like `invest in the stock market` or `pass my driving
test` should become practical first-session routes with stopping points, not
generic note-writing exercises. Finance and money-related prompts include a
safety note ending in `Not financial advice.`

Start a session:

```bash
curl -X POST https://unclump-api.onrender.com/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"task":"I need to reply to an email but I keep avoiding it","energy_level":"low"}'
```

Send feedback to keep the session moving:

```bash
curl -X POST https://unclump-api.onrender.com/api/session/{session_id}/feedback \
  -H "Content-Type: application/json" \
  -d '{"feedback":"need_smaller"}'
```

Ask for a contextual support nudge:

```bash
curl -X POST https://unclump-api.onrender.com/api/support \
  -H "Content-Type: application/json" \
  -d '{"task":"reply to emails","moment":"nudge","user_state":"overwhelmed"}'
```

Generate virtual coworking room messages:

```bash
curl -X POST https://unclump-api.onrender.com/api/coworking/chat \
  -H "Content-Type: application/json" \
  -d '{"task":"reply to emails","mode":"periodic","session_minutes":30,"coworkers":[{"name":"Maya","task":"sorting receipts","quirk":"uses_2"}]}'
```

Feedback values:

- `done`
- `need_smaller`
- `distracted`
- `skip`

Older clients may still send `too_hard` or `not_right`; the current app UI no
longer presents those controls.

## Architecture

```text
backend/
  engine.py       # Task breakdown + adaptive session engine
  main.py         # FastAPI server
  test_engine.py  # Engine tests
  test_api.py     # API tests

landing/
  index.html      # Landing page
  app.html        # Adaptive web app prototype
  logo.png        # Colour logo for light backgrounds
  logo_white.png  # White logo for dark backgrounds

docs/
  index.html      # GitHub Pages landing copy
  app.html        # GitHub Pages app copy
  trial.html      # Shareable private trial with virtual coworking
```

## Status

- [x] Task breakdown engine
- [x] Adaptive Unclump Session v2 with task sizing, goal interpretation, and checkpoints
- [x] DeepSeek/OpenAI provider adapter
- [x] Contextual support/nudge API
- [x] Virtual coworking sessions and chat API
- [x] FastAPI server
- [x] Landing page + web app prototype
- [x] Backend tests: 49 passing
- [ ] Persistent user accounts and profiles
- [ ] Real online body doubling rooms
- [ ] WhatsApp bot
- [ ] Flutter mobile app
