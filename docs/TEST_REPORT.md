# Verification report

Reviewed on 2026-11-19. These are archived snapshots, not verified working releases. Application code was preserved; failures below were not repaired as part of the branch split.

## Results

| Check | Baseline | Typing variant | Mood variant | Frontend snapshot |
| --- | --- | --- | --- | --- |
| Python source syntax | Pass | Pass | Pass, including `old/` | Pass |
| Import `main` in its existing environment | Fail: NumPy native extension missing | Pass | Fail: PyAudio native extension missing | Timed out after 30 seconds |
| Import `pi_api` in its existing environment | Pass | Pass | Fail: Pydantic Core native extension missing | Fail: FastAPI missing |
| Empty dashboard HTTP request, using shared working API environment | Pass | Pass | Fail: HTTP 500 | Pass |
| Dashboard reads synthetic logs emitted by its own logger | Fail in observed run | Fail in observed run | Pass | Fail: HTTP 500 |
| Preserve user role in conversations | Fail | Fail | Fail | Fail: HTTP 500 |
| CORS preflight from `http://localhost:5173` | Pass | Pass | Pass | Fail: HTTP 400 |
| Dashboard's message submission route | Fail: HTTP 404 | Fail: HTTP 404 | Fail: HTTP 404 | Fail: HTTP 404 |

All 54 Python files parsed successfully. Mood scalar mapping and CSV averaging passed checks for happy/sad/calm values, invalid labels, and an empty CSV.

The React dashboard production build passed. ESLint failed with three errors (`darkMode`, `Icon`, and `lastConnectionTime` unused) and one warning (the polling effect omits `fetchMoodData` from its dependencies). Vite warned that the JavaScript bundle exceeds 500 kB.

## What the API checks establish

`tools/verify_api.py` exercises actual FastAPI requests in process using HTTPX's ASGI transport. It uses temporary directories and synthetic conversations, without reading existing conversation logs or contacting the external response service. To separate application behavior from broken environments, all four API behavior checks ran using `b_bots(new)/venv3.11/bin/python` (FastAPI 0.109.0, Pydantic 1.10.11, HTTPX 0.28.1). This does not validate the mood variant against its originally installed Pydantic version.

The baseline and typing APIs read only the newest CSV of any kind. Their logger creates multiple CSVs, including a mood CSV, so assistant responses are not reliably selected. All returned conversations are marked `bot`, including user transcripts. The mood API also reads only the newest CSV and marks conversations as `bot`; its empty-state request tries to open today's user log before it exists. The frontend-side parser attempts to parse CSV header text as a timestamp and raises an error when reading its logger's output. Its CORS configuration contains a trailing slash in the allowed origin.

No supplied API implements `/generate/api/mood-message`. The frontend requests that endpoint and uses a relative `/generate` URL, while its Vite configuration has no API proxy. A successful production build alone does not verify these flows.

## Reproduce API checks

From the combined archive:

```bash
'b_bots(new)/venv3.11/bin/python' tools/verify_api.py --source b_bots
'b_bots(new)/venv3.11/bin/python' tools/verify_api.py --source 'b_bots(new)'
'b_bots(new)/venv3.11/bin/python' tools/verify_api.py --source b_bots_new_pkongkha
'b_bots(new)/venv3.11/bin/python' tools/verify_api.py --source b_bot_frontend
```

On a segment branch, install compatible FastAPI and HTTPX dependencies in a fresh test environment, then run `python tools/verify_api.py --source .`. Nonzero status indicates behavior failures; it is expected for the archived code.

## Not verified

Full microphone capture, wake-word accuracy, Vosk speech transcription, live mood inference accuracy, remote response generation, audio playback, and browser interaction were not verified. This execution environment has no `/dev/snd` audio device. Native dependencies in several existing environments are incomplete. A keyword-model silent-sample inference attempt timed out after 40 seconds without a result. The static audio route check did not complete within its timeout and is not counted as a pass. The `old/` and `full_main.py` alternatives received syntax checks only. Models and existing environments remain local and are excluded from Git.
