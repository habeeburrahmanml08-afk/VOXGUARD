"""Experimental laptop-microphone capture for speakerphone demonstrations."""

import argparse
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
INPUT_LENGTH = 64600


def check_audio_quality(audio, minimum_rms=0.003, clipping_threshold=0.99,
                        maximum_clipped_fraction=0.01):
    """Return measurable quality details and whether a window is usable."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size != INPUT_LENGTH:
        return False, {"reason": f"wrong sample count ({audio.size})"}
    if not np.isfinite(audio).all():
        return False, {"reason": "audio contains invalid numeric values"}

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    clipped_fraction = float(np.mean(np.abs(audio) >= clipping_threshold))
    details = {"rms": rms, "peak": peak, "clipped_fraction": clipped_fraction}
    if rms < minimum_rms:
        details["reason"] = "audio is too quiet"
        return False, details
    if clipped_fraction > maximum_clipped_fraction:
        details["reason"] = "audio is heavily clipped"
        return False, details
    return True, details


def capture_window(device=None):
    """Capture exactly one model-sized mono float32 window at 16 kHz."""
    return sd.rec(
        INPUT_LENGTH,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    ).reshape(-1)


def run_live(device=None):
    from predict import predict_audio_array

    window_seconds = INPUT_LENGTH / SAMPLE_RATE
    print("VOXGUARD live microphone demo")
    print(f"Capturing laptop microphone in {window_seconds:.2f}s windows at {SAMPLE_RATE} Hz.")
    print("For a cellular call demo, put the phone on speakerphone near the laptop mic.")
    print("Press Ctrl+C to stop. Call accuracy is unvalidated; results are experimental.")
    try:
        while True:
            audio = capture_window(device=device)
            usable, quality = check_audio_quality(audio)
            quality_summary = (
                f"RMS={quality.get('rms', float('nan')):.4f}, "
                f"peak={quality.get('peak', float('nan')):.3f}, "
                f"clipped={quality.get('clipped_fraction', float('nan')):.2%}"
            )
            if not usable:
                print(f"INSUFFICIENT AUDIO — {quality['reason']} ({quality_summary}); no assessment")
                continue

            result = predict_audio_array(audio)
            index = result["predicted_index"]
            mapped = {0: "POTENTIAL SPOOF", 1: "LIKELY BONA FIDE"}.get(index, "UNKNOWN CLASS")
            print(
                f"EXPERIMENTAL: {mapped} (class {index}; call accuracy unvalidated) | "
                f"inference={result['inference_ms']:.1f} ms | {quality_summary}"
            )
    except KeyboardInterrupt:
        print("\nMicrophone capture stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="show available microphone devices")
    parser.add_argument("--device", type=int, help="input device number from --list-devices")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    run_live(device=args.device)


if __name__ == "__main__":
    main()
