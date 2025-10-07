# listen_stt.py
import json, queue, sys, time
import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
MODEL_PATH = r".\models\vosk-model-small-en-us-0.15"

def transcribe_window(seconds=4):
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(bytes(indata))

    print(f"[listening {seconds}s] speak now…")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                           dtype='int16', channels=1, callback=cb):
        start = time.time()
        while time.time() - start < seconds:
            data = q.get()
            rec.AcceptWaveform(data)
        text = json.loads(rec.FinalResult()).get("text", "")
    print("[transcript]:", text)
    return text

if __name__ == "__main__":
    transcribe_window(4)
