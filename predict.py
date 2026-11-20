
import os
import sys
import time
from math import gcd
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

# Model configuration
MODEL_PATH = "models/aasist.onnx"
SAMPLE_RATE = 16000
INPUT_LENGTH = 64600

# Load model
print("Loading AASIST model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully!")


def predict_audio_array(audio):
    """Run one already-prepared mono float32 AASIST window."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size != INPUT_LENGTH:
        raise ValueError(f"Expected {INPUT_LENGTH} samples, got {audio.size}")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite values")

    model_input = audio.reshape(1, INPUT_LENGTH)
    started = time.perf_counter()
    outputs = session.run(None, {input_name: model_input})
    inference_ms = (time.perf_counter() - started) * 1000
    logits = np.asarray(outputs[0][0])
    predicted_index = int(np.argmax(logits))
    return {
        "logits": logits,
        "predicted_index": predicted_index,
        "inference_ms": inference_ms,
    }


def predict_audio(audio_path):

    if not os.path.exists(audio_path):
        print("ERROR: Audio file not found!")
        return

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != SAMPLE_RATE:
        divisor = gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(sr) // divisor,
        ).astype(np.float32)

    if len(audio) == 0:
        print("ERROR: Empty audio file!")
        return

    print("\nAudio loaded successfully")
    print("Duration:", round(len(audio) / SAMPLE_RATE, 2), "seconds")

    # Prepare audio length
    if len(audio) < INPUT_LENGTH:
        repeats = int(np.ceil(INPUT_LENGTH / len(audio)))
        audio = np.tile(audio, repeats)

    audio = audio[:INPUT_LENGTH]

    # Prepare model input. This preserves the existing recorded-audio behavior.
    audio = audio.astype(np.float32)

    # Input diagnostics
    print("Input RMS:", np.sqrt(np.mean(audio ** 2)))
    print("Input unique values:", len(np.unique(audio)))

    result = predict_audio_array(audio)
    logits = result["logits"]

    # Calculate relative scores
    exp_logits = np.exp(logits - np.max(logits))
    scores = exp_logits / np.sum(exp_logits)

    predicted_index = int(np.argmax(logits))

    print("\n========== VOXGUARD ==========")
    print("Logits:", logits)
    print("Class 0 relative softmax (not calibrated):", round(float(scores[0]), 4))
    print("Class 1 relative softmax (not calibrated):", round(float(scores[1]), 4))
    print("Predicted index:", predicted_index)
    print("Inference time:", round(result["inference_ms"], 2), "ms")

    if predicted_index == 0:
        print("Result: POTENTIAL SPOOF")
    elif predicted_index == 1:
        print("Result: LIKELY BONA FIDE")

    print("==============================")
    print("\nNOTE: This is an experimental prediction.")
    print("Phone-call accuracy and language coverage still need validation.")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py samples/fake.wav")
    else:
        predict_audio(sys.argv[1])

