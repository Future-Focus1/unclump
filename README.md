# Unclump

**Break through task paralysis.**

Unclump is an ADHD-friendly app that solves the "execution chasm" — the gap between knowing what to do and actually doing it.

## Live Endpoints

| Service | URL |
|---------|-----|
| 🌐 Landing page | https://3bbfef3c3aac510e-94-13-62-56.serveousercontent.com |
| 🔧 API (health) | https://39a564e2a78134da-94-13-62-56.serveousercontent.com/health |
| 🔧 API (docs) | https://39a564e2a78134da-94-13-62-56.serveousercontent.com/docs |

## Quick API Test

```bash
curl -X POST https://39a564e2a78134da-94-13-62-56.serveousercontent.com/api/breakdown \
  -H "Content-Type: application/json" \
  -d '{"task":"reply to emails"}'
```

## Architecture

```
backend/
├── engine.py       # Task breakdown engine (OpenAI + fallback)
├── main.py         # FastAPI server
├── test_engine.py  # 21 unit tests
└── test_api.py     # 7 integration tests

landing/
├── index.html      # Landing page
├── logo.png        # Colour logo (light backgrounds)
└── logo_white.png  # White logo (dark backgrounds)
```

## Status

- [x] Task breakdown engine (28 tests passing)
- [x] FastAPI server live (public URL)
- [x] Landing page live (public URL)
- [ ] Flutter mobile app
- [ ] WhatsApp bot
- [ ] Body doubling
