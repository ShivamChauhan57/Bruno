# bruno_wake_loop.py
import argparse, json, queue, time
import numpy as np
import requests
from jsonschema import validate, ValidationError

# ========= CONFIG =========
ESP32_IP = "192.168.0.144"
ESP32_BASE = f"http://{ESP32_IP}"
ESP32_COMMAND = f"{ESP32_BASE}/command"
MODEL_NAME = "phi3.5"           # Ollama tag
WHISPER_MODEL_SIZE = "small"    # "base" | "small" | "medium"
USE_GPU = False                 # True if you have CUDA
LANGUAGE = "en"

# Wake phrase options
WAKE_WORDS = ("hey bruno", "hi bruno", "okay bruno", "tommy", "charlie", "bruno")
WAKE_DEBOUNCE_S = 2.0           # ignore repeated wakes within this window

# Audio gating
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
ENERGY_THRESH = 100.0           # lower = more sensitive
UTT_MIN_S = 1.5
UTT_MAX_S = 8.0
SILENCE_HOLD_S = 0.7
# ==========================

BRUNO_SCHEMA = {
    "type": "object",
    "required": ["intent", "effects"],
    "properties": {
        "intent": {"type": "string"},
        "effects": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["part", "mode"],
                "properties": {
                    "part": {"type":"string","enum":[
                        "tail","head","left_ear","right_ear","left_eye","right_eye","chest","back","left_leg","right_leg"
                    ]},
                    "mode": {"type":"string","enum":["on","off","blink","pulse","wag","chase"]},
                    "hz": {"type":"number","minimum":0.1,"maximum":30},
                    "duty": {"type":"number","minimum":0.0,"maximum":1.0},
                    "duration_ms": {"type":"integer","minimum":50,"maximum":60000}
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are Bruno's Skill Planner, an AI system that thinks and reacts like Bruno — a loyal, playful robotic dog.
Your purpose is to interpret your master's spoken commands and translate them into **motor and LED actions**
that express how a real dog would behave in that situation.

Think like a dog first, then output a structured plan for how Bruno should move or react.

Your only output must be **valid JSON** (no explanations, comments, or extra text).

Follow this JSON schema exactly:
{
  "intent": "EMOTE_OR_ACTION",
  "effects": [
    { "part": "<one of: tail, head, left_ear, right_ear, left_eye, right_eye, chest, back, left_leg, right_leg>",
      "mode": "<on|off|blink|pulse|wag|chase>",
      "hz": <optional number>,
      "duty": <optional 0..1>,
      "duration_ms": <optional integer> }
  ]
}

Guidelines:
- "wag tail" -> {"part":"tail","mode":"wag","hz":6,"duration_ms":2000}
- "are you happy?" -> wag tail + eyes on (~2s)
- Use 1–3 effects. Default duration 1500 ms if unspecified.
- If unclear, default to tail wag 1.5s.
"""

FALLBACK_PLAN = {
    "intent": "EMOTE_OR_ACTION",
    "effects": [{"part":"tail","mode":"wag","hz":6,"duration_ms":1500}]
}

def ask_llm(user_text: str) -> dict:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_text}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=60
    )
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return FALLBACK_PLAN

def validate_or_fix(plan: dict) -> dict:
    try:
        validate(instance=plan, schema=BRUNO_SCHEMA)
        return plan
    except Exception:
        fixed = {"intent": plan.get("intent","EMOTE_OR_ACTION"), "effects": []}
        parts = {"tail","head","left_ear","right_ear","left_eye","right_eye","chest","back","left_leg","right_leg"}
        modes = {"on","off","blink","pulse","wag","chase"}
        for e in plan.get("effects", []):
            part = e.get("part","tail");   part = part if part in parts else "tail"
            mode = e.get("mode","wag");    mode = mode if mode in modes else "wag"
            hz   = float(e.get("hz", 6));  hz   = min(max(hz, 0.1), 30)
            duty = float(e.get("duty",0.5)); duty= min(max(duty,0.0), 1.0)
            dur  = int(e.get("duration_ms",1500)); dur = min(max(dur,50),60000)
            fixed["effects"].append({"part":part,"mode":mode,"hz":hz,"duty":duty,"duration_ms":dur})
        return fixed if fixed["effects"] else FALLBACK_PLAN

def send_to_esp32(plan: dict) -> str:
    r = requests.post(ESP32_COMMAND, json=plan, timeout=5)
    r.raise_for_status()
    return r.text

# ---------- audio + whisper ----------
def record_utterance(min_s=UTT_MIN_S, max_s=UTT_MAX_S, energy=ENERGY_THRESH,
                     frame_ms=FRAME_MS, silence_hold=SILENCE_HOLD_S):
    import sounddevice as sd
    q = queue.Queue()
    frame_len = int(SAMPLE_RATE * (frame_ms/1000.0))
    silence_frames_needed = int(silence_hold / (frame_ms/1000.0))
    buf = []
    silent_run = 0
    start = time.time()

    def cb(indata, frames, t, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                        callback=cb, blocksize=frame_len):
        while True:
            try:
                chunk = q.get(timeout=max_s)
            except queue.Empty:
                break
            x = chunk.reshape(-1)
            buf.append(x)
            rms = float(np.sqrt(np.mean(x**2)) * 32768.0)
            speaking = rms > energy

            if time.time() - start < min_s:
                pass
            else:
                silent_run = 0 if speaking else (silent_run + 1)
                if silent_run >= silence_frames_needed:
                    break

            if time.time() - start > max_s:
                break

    if not buf:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(buf, axis=0).astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / max(1.0, np.max(np.abs(audio)))
    return audio

def transcribe_audio(model, label=""):
    audio = record_utterance()
    if audio.size == 0:
        print(f"[{label}] (silence)")
        return ""
    segs, info = model.transcribe(
        audio, language=LANGUAGE, beam_size=5,
        condition_on_previous_text=False, temperature=0.0, vad_filter=False
    )
    text = "".join(s.text for s in segs).strip().lower()
    print(f"[{label}] {text}")
    return text

def contains_wake(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in WAKE_WORDS)

def strip_wake(text: str) -> str:
    t = text.lower()
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip(" ,.?!")

# ---------- main loop ----------
def main():
    import sounddevice as sd
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser(description="Continuous wake-word listener for Bruno")
    ap.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    ap.add_argument("--energy", type=float, default=ENERGY_THRESH, help="Energy threshold")
    args = ap.parse_args()

    global USE_GPU
    if args.gpu: USE_GPU = True

    print(f"ESP32: {ESP32_BASE}  ->  /command")
    print("Loading Whisper…")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cuda" if USE_GPU else "cpu",
        compute_type="float16" if USE_GPU else "int8"
    )
    print("Ready. Say 'hey bruno ...'   (Ctrl+C to exit)")
    last_wake_ts = 0.0

    try:
        while True:
            # --- Wake phase ---
            print("🎧  listening for wake…")
            wake_text = transcribe_audio(model, label="wake")
            if not wake_text:
                continue
            if not contains_wake(wake_text):
                continue
            if time.time() - last_wake_ts < WAKE_DEBOUNCE_S:
                # ignore rapid double-triggers
                continue
            last_wake_ts = time.time()

            # Command can be in the same utterance (e.g., "hey bruno wag your tail")
            cmd_inline = strip_wake(wake_text)
            if len(cmd_inline.split()) >= 2:
                cmd = cmd_inline
            else:
                # --- Command phase ---
                print("🗣️  wake heard. say the command…")
                cmd = transcribe_audio(model, label="cmd")

            cmd = strip_wake(cmd)
            if not cmd:
                cmd = "wag your tail"

            print(f"[command] {cmd}")

            # LLM → JSON
            plan = ask_llm(cmd)
            plan = validate_or_fix(plan)
            print("[plan]", json.dumps(plan, indent=2))

            # Send to ESP32
            try:
                resp = send_to_esp32(plan)
                print("[esp32]", resp)
            except Exception as e:
                print("[esp32 error]", e)

            # brief pause before re-arming wake
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nbye!")

if __name__ == "__main__":
    main()
