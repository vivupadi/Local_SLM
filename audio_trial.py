import soundfile as sf
import sounddevice as sd
import numpy as np

INPUT_DEVICE = 2
OUTPUT_DEVICE= 0

SAMPLE_RATE= 16000
DURATION= 5
CHANNELS=2

print("Recording for 5 seconds...speak now")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    device=INPUT_DEVICE,
    dtype='int16'
)

sd.wait()
print("Recording done!")

#save the file
sf.write('test_python.wav', audio, SAMPLE_RATE)
print("File saved to test_python.wav")

#Playback
print("Playing the recording")
data,fs = sf.read('test_python.wav')
sd.play(data, fs, device=OUTPUT_DEVICE)
sd.wait()
print("Playback done")