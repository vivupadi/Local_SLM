import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# Load model (downloads ~145MB on first run)
print("⏳ Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Model loaded!")

# Audio settings
SAMPLE_RATE = 16000
DURATION = 5
INPUT_DEVICE = 2  # seeed mic

print("\n🎙️  Speak for 5 seconds...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    device=INPUT_DEVICE,
    dtype='int16'
)
sd.wait()
print("✅ Recording done!")

# Save temporarily
sf.write('stt_test.wav', audio, SAMPLE_RATE)

# Transcribe
print("⏳ Transcribing...")
segments, info = model.transcribe(
    'stt_test.wav',
    language="en",
    beam_size=5
)

print("\n📝 Transcription:")
for segment in segments:
    print(f"  {segment.text}")
