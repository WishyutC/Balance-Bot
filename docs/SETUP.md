# Setup and operation

## Python assistant

Use a separate environment for each variant. Run commands from the selected variant directory: imports, model paths, and output paths assume that working directory.

```bash
cd b_bots  # or 'b_bots(new)', b_bots_new_pkongkha, b_bot_frontend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p logs audio_pcm audio_output output_realtime
python main.py
```

The legacy manifests include platform-sensitive packages (TensorFlow, TFLite runtime, PyAudio, and TTS). Choose a compatible Python/platform combination; installation has not been verified. PortAudio development libraries and FFmpeg may be needed for audio input/output. The frontend-side Python requirements omit FastAPI, Uvicorn, and Pydub even though its modules import them. The mood variant's manifest contains `tensorflows`, an apparent package-name error; resolve and validate that environment before installing. Original manifests are preserved as historical evidence.

## Models and remote service

Obtain the original trained models separately and put them under `models/` in the selected variant:

- `keyword_model.tflite`: keyword inference model.
- `vosk-model-en-us-0.22/`: extracted Vosk speech model directory.
- Mood variant only: the Keras model named by `MOOD_MODEL` in `config.py` (currently `2025-10-05_01.40.46 A91_L0.28_TA90_TL0.28_VA72_VL0.87 mood-recognition.keras`).

Review `API_URL`, audio settings, thresholds, and model paths in the selected `config.py`. The response service is external and is not started by this repository. Model provenance and download locations are not supplied here; Git clones do not include weights.

## Dashboard API

In a second terminal, activate the same variant's environment and run from its directory:

```bash
mkdir -p logs audio_pcm
python pi_api.py
```

The APIs use port 3000 and expose `GET /generate`. The mood API also mounts `/audio_pcm`; that directory must exist at startup. Review CORS settings for the intended frontend origin. The frontend-side API currently specifies `http://localhost:5173/` with a trailing slash, which may not match the browser's origin.

## React dashboard

```bash
cd b_bot_frontend/b_bot_app
npm ci
npm run dev
```

Use `npm run build` for a production build and `npm run lint` for lint checks. Configure a development proxy or same-origin reverse proxy to route `/generate` (and `/audio_pcm` when used) to the intended Python API. No such proxy is included. The mood-message write route requested by the dashboard is missing from these APIs; that feature needs implementation before use.

## Verification limits

Source syntax can be checked without installing the machine-learning stack. Full operation additionally requires the original models, audio devices, external response service, and a resolved dependency environment. A successful syntax check does not establish runtime compatibility or working dashboard integration.
