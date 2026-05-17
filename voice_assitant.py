import pyaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import ollama
import subprocess
import tempfile
import os
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ─────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────
INPUT_DEVICE = 2          # seeed mic
OUTPUT_DEVICE = 0         # Pi 3.5mm jack
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
PIPER_MODEL = os.path.expanduser("~/piper_models/en_US-amy-medium.onnx")
LLM_MODEL = "llama3.2:1b"

# Conversation history (context memory)
conversation_history = [
    {
        "role": "system",
        "content": "You are Jarvis, a helpful voice assistant. Keep responses short and conversational — maximum 2-3 sentences. You are running locally on a Raspberry Pi."
    }
]

# ─────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────
print("⏳ Loading wake word model...")
wake_model = WakeWordModel(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

print("⏳ Loading Whisper STT model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")

print("✅ All models loaded! Say 'Hey Jarvis' to start.\n")

# ─────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────
def record_audio(seconds=RECORD_SECONDS):
    """Record audio from ReSpeaker mic"""
    print(f"🎙️  Listening for {seconds} seconds...")
    frames = []
    num_chunks=int(SAMPLE_RATE /CHUNK * seconds)
    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    audio_bytes=b''.join(frames)
    audio_np=np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_np.reshape(-1,1)
    """audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        device=INPUT_DEVICE,
        dtype='int16'
    )
    sd.wait()
    return audio"""

def transcribe(audio):
    """Convert audio to text using Whisper"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    segments, _ = stt_model.transcribe(
        tmp.name,
        language="en",
        beam_size=5
    )
    os.unlink(tmp.name)
    text = " ".join([s.text for s in segments]).strip()
    return text

def ask_llm(user_text):
    """Send text to Ollama LLM and get response"""
    conversation_history.append({
        "role": "user",
        "content": user_text
    })
    response = ollama.chat(
        model=LLM_MODEL,
        messages=conversation_history,
        options={"num_ctx": 2048}
    )
    reply = response['message']['content']
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })
    return reply

def speak(text):
    """Convert text to speech using Piper and play it"""
    print(f"🔊 Jarvis: {text}")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    subprocess.run([
        "python3", "-m", "piper",
        "--model", PIPER_MODEL,
        "--output_file", tmp.name
    ], input=text.encode(), capture_output=True)
    data, fs = sf.read(tmp.name)
    sd.play(data, fs, device=OUTPUT_DEVICE)
    sd.wait()
    os.unlink(tmp.name)

# ─────────────────────────────────────
# WAKE WORD DETECTION LOOP
# ─────────────────────────────────────
CHUNK = 1280
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=INPUT_DEVICE
)

try:
    while True:
        # Listen for wake word
        audio_chunk = np.frombuffer(
            stream.read(CHUNK, exception_on_overflow=False),
            dtype=np.int16
        )
        prediction = wake_model.predict(audio_chunk)

        for key, value in prediction.items():
            if value > 0.5:
                print(f"\n✅ Wake word detected! ({value:.2f})")
                #stream.stop_stream()

                # Record question
                audio = record_audio(seconds=5)

                # Transcribe
                print("⏳ Transcribing...")
                user_text = transcribe(audio)
                if not user_text:
                    print("❌ Couldn't hear anything, try again.")
                    #stream.start_stream()
                    continue
                print(f"👤 You: {user_text}")

                # Get LLM response
                print("⏳ Thinking...")
                reply = ask_llm(user_text)

                # Speak response
                print("Speaking soon....")
                speak(reply)

                # Resume listening
                #stream.start_stream()
                print("\n👂 Listening for 'Hey Jarvis'...")

except KeyboardInterrupt:
    print("\n🛑 Shutting down Jarvis...")
    #stream.stop_stream()
    stream.close()
    p.terminate()
