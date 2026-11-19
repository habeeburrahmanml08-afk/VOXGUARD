
import numpy as np
import soundfile as sf
import onnxruntime as ort

MODEL_PATH = "models/aasist.onnx"
AUDIO_PATH = "samples/reference_human.wav"

# Read audio
audio, sr = sf.read(AUDIO_PATH, dtype="float32")

if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

if sr != 16000:
    raise ValueError("Audio must be sampled at 16000 Hz")

if len(audio) == 0:
    raise ValueError("Audio is empty")

# Prepare the required 64600-sample input
target_length = 64600

if len(audio) < target_length:
    repeats = target_length // len(audio) + 1
    audio = np.tile(audio, repeats)

audio = audio[:target_length].astype(np.float32)
audio = audio.reshape(1, -1)

# Load model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# Run inference
result = session.run(
    ["logits"],
    {"wav": audio}
)[0][0]

print("\nVOXGUARD - MODEL TEST")
print("-------------------------")
print("Raw logits:", result.tolist())
print("Winning output index:", int(np.argmax(result)))
print("-------------------------")
