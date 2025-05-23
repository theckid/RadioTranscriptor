import whisper
import sounddevice as sd
import numpy as np
import threading
from datetime import datetime
import scipy.signal
import time
from collections import deque

# Load Whisper model on GPU (if available)
import torch

model = whisper.load_model("medium").to("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Whisper is using: {model.device}")

# Audio config
samplerate = 48000
block_duration = 0.5  # seconds
samples_per_block = int(samplerate * block_duration)
buffer_duration = 10  # seconds for rolling buffer
samples_per_buffer = int(samplerate * buffer_duration)

# VAD threshold
vad_energy_threshold = 0.01

# Rolling buffer to store the last 10 seconds
rolling_buffer = deque(maxlen=samples_per_buffer)
speech_active = False
speech_timer = 0
speech_timeout = 2  # seconds of silence before stopping transcription window

# Buffer to store active voice session
active_speech_buffer = []

# Lock for thread-safe buffer access
buffer_lock = threading.Lock()

def audioplayback(audio_data):
    audio_data = audio_data.astype(np.float32)
    normalize_audio = audio_data / np.max(np.abs(audio_data))
    print("Playing back audio data...")
    sd.play(normalize_audio, samplerate=48000)
    sd.wait()

def transcribe_buffer(buffer):
    audio_data = np.array(buffer)

    if np.max(np.abs(audio_data)) == 0:
        print("🛑 Skipping empty buffer")
        return

    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))

    # audioplayback(audio_data)

    # Resample from 48kHz to 16kHz
    resampled = scipy.signal.resample_poly(audio_data, up=1, down=3)

    try:
        result = model.transcribe(resampled.astype(np.float32), language='en')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [RADIO] {result['text']}")

        log_filename = f"radio_transcript_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {result['text']}\n")
    except Exception as e:
        print(f"❌ Transcription error: {e}")


def monitor_speech():
    global speech_active, speech_timer, active_speech_buffer
    while True:
        if speech_active:
            if time.time() - speech_timer > speech_timeout:
                print("🛑 Voice ended — transcribing...")
                speech_active = False
                with buffer_lock:
                    captured = active_speech_buffer.copy()
                    active_speech_buffer = []
                transcribe_buffer(captured)
        time.sleep(0.5)


def callback(indata, frames, time_info, status):
    global speech_active, speech_timer, active_speech_buffer
    if status:
        print("⚠️", status)

    audio_chunk = indata[:, 0]  # Mono
    with buffer_lock:
        rolling_buffer.extend(audio_chunk)

    # Voice Activity Detection (RMS Energy)
    energy = np.sqrt(np.mean(audio_chunk**2))
    if energy > vad_energy_threshold:
        if not speech_active:
            print("🎤 Voice started...")
            with buffer_lock:
                active_speech_buffer = list(rolling_buffer)  # capture pre-roll
        speech_active = True
        speech_timer = time.time()
        with buffer_lock:
            active_speech_buffer.extend(audio_chunk)
    elif speech_active:
        # Still in active window but no voice this block
        with buffer_lock:
            active_speech_buffer.extend(audio_chunk)


# Start the stream and monitoring thread
threading.Thread(target=monitor_speech, daemon=True).start()

with sd.InputStream(device=1, channels=1, samplerate=samplerate,
                    callback=callback, blocksize=samples_per_block,
                    dtype='float32'):
    print("🎧 Listening for voice... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopped.")
