# Version inventory

## Baseline: `b_bots`

`main.py` coordinates keyword detection, speech recognition, logging, and audio playback. `full_main.py` preserves a standalone implementation; it is a historical alternative, not an additional service to launch. `pi_api.py` exposes dashboard data through `GET /generate` on port 3000.

## Typing variant: `b_bots(new)`

A source comparison of the top-level files shows that only `main.py` differs from `b_bots`. It adds a mode menu and sends typed messages to the remote service. The constructor still initializes assistant components, so typing mode does not guarantee operation without the models or audio dependencies. The duplicated modules are retained to keep this historical snapshot runnable in its original layout.

## Mood variant: `b_bots_new_pkongkha`

Adds `emotion.py`, `mood_recognizer.py`, and `mood_parser.py`. Mood inference uses Keras with the JAX backend. Keyword inference uses `ai_edge_litert`, with a 0.80 threshold and four detection slices, compared with the baseline's 0.90 threshold. `pi_api.py` adds mood data and serves WAV files from `/audio_pcm`.

The `old/` subdirectory is an earlier snapshot, preserved for comparison. Its files and requirements are separate from the top-level variant; do not mix their modules.

## Dashboard: `b_bot_frontend`

`b_bot_app/` is a React 19 application built with Vite, Tailwind CSS, Recharts, and Lucide icons. The adjacent Python files are another assistant/API snapshot. Its `config.py` writes audio to `output_realtime`.

The dashboard fetches `/generate` relative to its web origin. The checked-in Vite configuration does not proxy this route. The dashboard also requests `/generate/api/mood-message`, which the supplied Python APIs do not implement. Integration needs additional routing/backend work.

## Organization policy

Keep the four original variants and the `old/` snapshot on this archival branch. Do not silently merge implementations or delete historical source based on folder names. Generated files and environments are excluded instead of uploaded. Select and validate a maintained variant before a future consolidation.
