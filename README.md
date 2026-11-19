# Balance Bot

A voice assistant with keyword detection, Vosk speech recognition, a remote response/audio service, and a React mood dashboard. This repository collects four existing project directories. It preserves their names because the Python programs use local imports and paths relative to the working directory.

## Choose a version

| Directory | Role | Entry points |
| --- | --- | --- |
| [`b_bots`](b_bots/) | Baseline modular voice assistant; includes an older standalone implementation | `main.py`, `pi_api.py`; legacy `full_main.py` |
| [`b_bots(new)`](b_bots%28new%29/) | Baseline plus interactive voice/typing selection | `main.py`, `pi_api.py` |
| [`b_bots_new_pkongkha`](b_bots_new_pkongkha/) | Mood recognition with Keras/JAX, LiteRT keyword inference, mood-aware dashboard API | `main.py`, `pi_api.py` |
| [`b_bot_frontend`](b_bot_frontend/) | React/Vite dashboard and an accompanying earlier Python assistant/API | `b_bot_app/`, `main.py`, `pi_api.py` |

Start with [setup instructions](docs/SETUP.md), then consult the [version inventory](docs/VERSIONS.md) for differences and known limitations. No release numbers or production status have been inferred from folder names.

## Repository contents

Only application source, dependency manifests, frontend assets, and documentation belong in Git. Virtual environments, installed packages, model weights, logs, conversation audio, build output, and the frontend tar archive are excluded by `.gitignore`. Existing local copies are preserved.

The remote response service is configured in each variant's `config.py`; its implementation is not included here. Running the application requires model files, audio hardware, compatible dependencies, and access to that service. The archived variants have not been validated end to end.

## Separate branches

Each segment is also available on its own branch with its files at the repository root. See the [branch map](docs/BRANCHES.md) and [verification report](docs/TEST_REPORT.md). The API behavior checks found failures; these are archival snapshots, not verified working releases.
