"""Small local HTTP API for VOXGUARD recorded audio and laptop-mic demos."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

from live_microphone import INPUT_LENGTH, SAMPLE_RATE, capture_window, check_audio_quality
from predict import predict_audio_array


HOST = "127.0.0.1"
PORT = 8000
API_VERSION = "1.1.0"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
AGGREGATION_WINDOWS = 3

_lock = threading.Lock()
_live_stop = threading.Event()
_live_thread = None
_live_state = {
    "running": False,
    "windows_processed": 0,
    "latest": None,
    "aggregate": {
        "label": "COLLECTING AUDIO",
        "class_index": None,
        "assessment": "waiting for 3 consecutive quality-checked windows",
        "windows_remaining": AGGREGATION_WINDOWS,
    },
    "error": None,
}


def _label_for(class_index):
    label = {0: "POTENTIAL SPOOF", 1: "LIKELY BONA FIDE"}.get(class_index, "UNKNOWN CLASS")
    return {
        "label": label,
        "class_index": class_index,
        "assessment": "experimental; not a verified fraud verdict",
    }


def _aggregate_recent_classes(recent_classes):
    if len(recent_classes) < AGGREGATION_WINDOWS:
        return {
            "label": "COLLECTING AUDIO",
            "class_index": None,
            "assessment": "waiting for 3 consecutive quality-checked windows",
            "windows_remaining": AGGREGATION_WINDOWS - len(recent_classes),
        }
    spoof_votes = recent_classes.count(0)
    bona_fide_votes = recent_classes.count(1)
    winning_class = 0 if spoof_votes > bona_fide_votes else 1
    return {
        **_label_for(winning_class),
        "method": "majority vote over the latest 3 consecutive quality-checked windows",
        "windows_used": AGGREGATION_WINDOWS,
        "votes": {"potential_spoof": spoof_votes, "likely_bona_fide": bona_fide_votes},
    }


def _prepare_uploaded_audio(payload):
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("The audio file is empty")
    if not np.isfinite(audio).all():
        raise ValueError("The audio contains invalid numeric values")
    original_duration = float(audio.size / sample_rate)
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    if audio.size < INPUT_LENGTH:
        raise ValueError(
            f"Need at least {INPUT_LENGTH / SAMPLE_RATE:.2f} seconds of audio; "
            f"received {original_duration:.2f} seconds"
        )
    window = np.asarray(audio[:INPUT_LENGTH], dtype=np.float32)
    usable, quality = check_audio_quality(window)
    return window, quality, usable, original_duration


def _live_worker(device):
    global _live_state
    recent_classes = []
    try:
        while not _live_stop.is_set():
            audio = capture_window(device=device)
            usable, quality = check_audio_quality(audio)
            if usable:
                prediction = predict_audio_array(audio)
                class_index = int(prediction["predicted_index"])
                result = {
                    **_label_for(class_index),
                    "inference_ms": round(float(prediction["inference_ms"]), 2),
                }
                recent_classes.append(class_index)
                if len(recent_classes) > AGGREGATION_WINDOWS:
                    recent_classes.pop(0)
                aggregate = _aggregate_recent_classes(recent_classes)
            else:
                recent_classes.clear()
                result = {
                    "label": "INSUFFICIENT AUDIO",
                    "class_index": None,
                    "assessment": "no model assessment made",
                    "reason": quality.get("reason"),
                    "inference_ms": None,
                }
                aggregate = {
                    "label": "INSUFFICIENT AUDIO",
                    "class_index": None,
                    "assessment": "audio-quality failure reset the combined assessment",
                    "reason": quality.get("reason"),
                }
            result["audio_quality"] = quality
            result["captured_at_unix"] = time.time()
            with _lock:
                _live_state["windows_processed"] += 1
                _live_state["latest"] = result
                _live_state["aggregate"] = aggregate
    except Exception as exc:
        with _lock:
            _live_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _live_state["running"] = False


class VoxguardHandler(BaseHTTPRequestHandler):
    server_version = "VOXGUARD-local/1.0"

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {
                "ok": True,
                "service": "VOXGUARD",
                "api_version": API_VERSION,
                "model_loaded": True,
                "sample_rate": SAMPLE_RATE,
                "window_samples": INPUT_LENGTH,
                "window_seconds": round(INPUT_LENGTH / SAMPLE_RATE, 4),
                "aggregation_windows": AGGREGATION_WINDOWS,
            })
            return
        if path == "/api/devices":
            devices = []
            for index, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    devices.append({"index": index, "name": device["name"], "input_channels": device["max_input_channels"]})
            self._send_json(200, {"devices": devices})
            return
        if path == "/api/live/status":
            with _lock:
                state = dict(_live_state)
            self._send_json(200, state)
            return
        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        global _live_thread, _live_state
        path = urlparse(self.path).path
        if path == "/api/predict":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._send_json(413 if length > MAX_UPLOAD_BYTES else 400, {"error": "Upload must be between 1 byte and 30 MB"})
                return
            try:
                window, quality, usable, duration = _prepare_uploaded_audio(self.rfile.read(length))
            except (RuntimeError, ValueError, sf.SoundFileError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            if not usable:
                self._send_json(422, {"label": "INSUFFICIENT AUDIO", "reason": quality.get("reason"), "audio_quality": quality})
                return
            try:
                prediction = predict_audio_array(window)
            except Exception as exc:
                self._send_json(500, {"error": f"Model inference failed: {type(exc).__name__}: {exc}"})
                return
            self._send_json(200, {
                **_label_for(prediction["predicted_index"]),
                "inference_ms": round(float(prediction["inference_ms"]), 2),
                "audio_duration_seconds": round(duration, 3),
                "audio_quality": quality,
                "window_samples": INPUT_LENGTH,
                "sample_rate": SAMPLE_RATE,
            })
            return

        if path == "/api/live/start":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 4096:
                    raise ValueError("Request body is too large")
                request = json.loads(self.rfile.read(length) or b"{}")
                device = request.get("device")
                if device is not None:
                    device = int(device)
                    info = sd.query_devices(device)
                    if info["max_input_channels"] < 1:
                        raise ValueError("Selected device is not an input microphone")
            except (ValueError, TypeError, json.JSONDecodeError, sd.PortAudioError) as exc:
                self._send_json(400, {"error": f"Invalid microphone device: {exc}"})
                return
            with _lock:
                if _live_state["running"]:
                    self._send_json(409, {"error": "Live capture is already running"})
                    return
                _live_stop.clear()
                _live_state = {
                    "running": True,
                    "windows_processed": 0,
                    "latest": None,
                    "aggregate": _aggregate_recent_classes([]),
                    "error": None,
                }
                _live_thread = threading.Thread(target=_live_worker, args=(device,), daemon=True)
                _live_thread.start()
            self._send_json(202, {"running": True, "message": "Live microphone capture started"})
            return

        if path == "/api/live/stop":
            _live_stop.set()
            with _lock:
                running = _live_state["running"]
            self._send_json(200, {"running": running, "message": "Stop requested"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    print(f"Loading VOXGUARD AASIST model; API will use http://{HOST}:{PORT}")
    try:
        server = ThreadingHTTPServer((HOST, PORT), VoxguardHandler)
    except OSError as exc:
        raise SystemExit(
            f"Could not start VOXGUARD API on {HOST}:{PORT}. "
            f"The port may already be in use. Details: {exc}"
        )
    print("API ready. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VOXGUARD API...")
    finally:
        _live_stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
