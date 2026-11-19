
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

sample_rate = 16000
duration = 8

print("Recording will start now...")
print("Speak normally for 5 seconds.")

try:
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocking=True
    )

    Path("samples").mkdir(exist_ok=True)

    sf.write(
        "samples/human.wav",
        recording,
        sample_rate,
        subtype="PCM_16"
    )

    print("\nRecording saved successfully!")

    info = sf.info("samples/human.wav")
    print("Duration:", info.duration, "seconds")
    print("Sample rate:", info.samplerate)
    print("Channels:", info.channels)
    print("File size:", Path("samples/human.wav").stat().st_size)

except Exception as e:
    print("Recording failed:", e)
