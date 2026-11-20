# VOXGUARD

**AI-Powered Voice Deepfake Detection System**

## About
VOXGUARD is a prototype designed to analyse uploaded audio and identify whether it is likely to be genuine speech or a potential AI-generated voice spoof.

## Proposed Workflow
1. Provide a WAV file or laptop-microphone audio.
2. Preprocess it to the AASIST input format and check audio quality.
3. Analyse one window using the AASIST ONNX model.
4. Return an experimental assessment through the local API.

## Technology Stack
- Python
- PyTorch
- ONNX Runtime
- Librosa
- SoundFile
- Local Python HTTP API (`backend_api.py`)
- HTML, CSS and JavaScript (planned)

## Current Progress
- [x] Python environment setup
- [x] AASIST ONNX model loading
- [x] Initial audio recording and inference tests
- [ ] Prediction accuracy validation on a representative labeled dataset
- [x] Local prediction API for WAV uploads and laptop microphone capture
- [ ] Frontend development
- [ ] End-to-end testing

## Important Note
VOXGUARD is a prototype. The output mapping used here is class 0 = potential spoof and class 1 = likely bona fide; this matches the model card's higher-is-more-bona-fide description and the reference sample outputs. Model accuracy on phone-speaker audio and Tamil-English speech has not been validated. Treat predictions as experimental indications, not proof of authenticity. The [Hugging Face model card](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST) reports 43.01% EER on InTheWild, so benchmark performance should not be assumed to transfer to real calls.

## Experimental live microphone demo

The live demo captures audio from a Windows laptop microphone. To demonstrate a cellular call, put the Android phone on speakerphone near the laptop microphone. This is acoustic capture and does not intercept the cellular call audio stream.

From PowerShell, open the VOXGUARD project folder and run:

```powershell
cd D:\VOXGUARD
.\venv\Scripts\Activate.ps1
python live_microphone.py --list-devices
python live_microphone.py
```

If Windows blocks script activation, run the project interpreter directly:

```powershell
cd D:\VOXGUARD
.\venv\Scripts\python.exe live_microphone.py --list-devices
.\venv\Scripts\python.exe live_microphone.py
```

Use `--device NUMBER` to select the laptop microphone shown in the device list. Each capture is one non-overlapping 64,600-sample window (about 4.04 seconds) at 16 kHz, followed by one AASIST inference. The output includes measured model inference time and basic RMS, peak, and clipping checks. Low-volume or heavily clipped windows show `INSUFFICIENT AUDIO` and do not produce a model label. Press Ctrl+C to stop.

Live labels are experimental indications only. No calibrated probability is reported. A single recorded sample run on the development laptop measured 182.17 ms for the ONNX `session.run` call; this excludes model startup, audio capture, and preprocessing, and is not a benchmark guarantee. The separate recorded-audio command remains available:

```powershell
python predict.py samples\fake.wav
```

## Local backend API

The API uses Python's built-in HTTP server and the packages already in the project environment; it does not require FastAPI or an internet connection. It listens only on this laptop (`127.0.0.1`). A frontend on the same laptop can send a WAV `File` as the raw request body to `POST /api/predict` with `Content-Type: audio/wav`. The API also controls the same laptop microphone used by the command-line demo.

In PowerShell, start the API in the project folder:

```powershell
cd D:\VOXGUARD
.\venv\Scripts\python.exe backend_api.py
```

Keep that window open. In a second PowerShell window, check that the service and microphone are available:

```powershell
cd D:\VOXGUARD
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/devices
```

To classify a WAV file of at least 4.04 seconds:

```powershell
curl.exe -X POST -H "Content-Type: audio/wav" --data-binary "@D:\VOXGUARD\samples\fake.wav" http://127.0.0.1:8000/api/predict
```

To start microphone capture with input device 1, check the latest result, and stop capture:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/live/start -Method Post -ContentType 'application/json' -Body '{"device":1}'
Start-Sleep -Seconds 6
Invoke-RestMethod http://127.0.0.1:8000/api/live/status
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/live/stop -Method Post
```

Endpoints: `GET /api/health`, `GET /api/devices`, `POST /api/predict` (raw WAV request body), `POST /api/live/start` (optional JSON `{"device":1}`), `GET /api/live/status`, and `POST /api/live/stop`. Live status provides the latest per-window result and an experimental rolling majority vote over the latest three consecutive quality-checked windows. Three windows take about 12.1 seconds of capture plus inference time. A poor-quality window resets the vote. Results include class index, audio-quality measurements, and measured inference time, but no calibrated probabilities. Live call audio is captured acoustically from the laptop microphone while the phone is on speakerphone; the API does not intercept cellular-call audio. Multilingual accuracy has not been validated.

## Validate on labeled WAV files

Before describing model accuracy, evaluate recordings with known ground truth. Create these folders and put only correctly labeled WAV files inside. The following two-file example checks that the evaluation flow runs; it is far too small to establish model accuracy:

```text
validation\bonafide\
validation\spoof\
```

Quick pipeline check with the two provided reference clips:

```powershell
cd D:\VOXGUARD
New-Item -ItemType Directory -Force validation\bonafide, validation\spoof
Copy-Item samples\reference_human.wav validation\bonafide\
Copy-Item samples\fake.wav validation\spoof\
.\venv\Scripts\python.exe evaluate_validation.py --dataset validation
```

The JSON report includes coverage, accuracy among scored files, bona fide false-alarm rate, spoof miss rate, per-file results, and mean inference time. Exact duplicate audio files are excluded from the metrics. For meaningful validation, add many independent, representative phone-speaker recordings with verified labels, including both Tamil-English and other relevant speech. Do not use filenames as ground truth unless the audio provenance is known.

## Team
Six-member development team.
