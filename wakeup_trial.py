import pyaudio
import numpy as np
from openwakeword.model import Model

#Load model
model= Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

#Audio settings
CHUNK=1280
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000

p =pyaudio.PyAudio()
stream=p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input= True,
    frames_per_buffer=CHUNK,
    input_device_index=2
)

print("Listenin for Hey Jarvis")

try:
    while True:
        audio_chunk=np.frombuffer(
            stream.read(CHUNK),
            dtype=np.int16
        )
        prediction=model.predict(audio_chunk)
        for key,value in prediction.items():
            if value > 0.5:
                print(f"Wake word detected! {key}: {value:.2f}")
except KeyboardInterrupt:
    print("\n stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()