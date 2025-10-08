# bruno_wake_loop_vosk.py
import argparse, json, queue, time
import numpy as np
import requests
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from jsonschema import validate, ValidationError

# ========= CONFIG =========
ESP32_IP = "192.168.0.144"
ESP32_BASE = f"http://{ESP32_IP}"
ESP32_COMMAND = f"{ESP32_BASE}/command"
MODEL_NAME = "phi3.5"           # Ollama tag

# Vosk
VOSK_MODEL_PATH = r".\models\vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
CHANNELS = 1
LISTEN_WAKE_S = 3.0             # window to catch wake
LISTEN_CMD_S  = 4.5             # window to catch command
FRAME_BYTES = int(SAMPLE_RATE * 0.02) * 2  # 20ms frames * int16 mono

# Wake phrase options (matches your Whisper version)
WAKE_WORDS = ("hey bruno", "hi bruno", "okay bruno", "tommy", "charlie", "bruno")
WAKE_DEBOUNCE_S = 2.0           # ignore repeated wakes within this window
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

# ----- LLM (Ollama) -----
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

# ----- Vosk helpers -----
def _stream_text(rec: KaldiRecognizer, seconds: float) -> str:
    """
    Capture audio for ~seconds, feed recognizer in ~20ms frames, return final text.
    """
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_BYTES//2,  # ~10ms blocks
                           dtype='int16', channels=CHANNELS, callback=cb):
        start = time.time()
        while time.time() - start < seconds:
            try:
                data = q.get(timeout=seconds)
            except queue.Empty:
                break
            rec.AcceptWaveform(data)

    try:
        result = json.loads(rec.FinalResult())
        text = result.get("text","").strip().lower()
    except Exception:
        text = ""
    return text

def hear_wake(model: Model) -> bool:
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    # Grammar drastically speeds up & improves matching for short wake phrases
    grammar = json.dumps(list(WAKE_WORDS) + ["[unk]"])
    rec.SetGrammar(grammar)
    rec.SetWords(True)
    print("🎧 listening for wake…")
    text = _stream_text(rec, LISTEN_WAKE_S)
    print("[wake]", text or "(silence)")
    return any(w in text for w in WAKE_WORDS)

def hear_command(model: Model) -> str:
    # Start with a light domain grammar to bias likely words
    domain = [
        "wag your tail","blink your eyes","turn on head","turn off head",
        "are you happy","happy","sad","fast","slow",
        "tail","head","left ear","right ear","left eye","right eye","chest","back","left leg","right leg",
        "blink","pulse","wag","chase","on","off"
    ]
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    rec.SetGrammar(json.dumps(domain + ["[unk]"]))
    print("🗣️  say the command…")
    text = _stream_text(rec, LISTEN_CMD_S)

    # If grammar result is too short/unsure, re-run free-form
    if len(text.split()) < 2:
        rec2 = KaldiRecognizer(model, SAMPLE_RATE)
        rec2.SetWords(True)
        print("…retrying free-form")
        text2 = _stream_text(rec2, LISTEN_CMD_S)
        if len(text2) > len(text):
            text = text2

    print("[cmd]", text or "(silence)")
    return text

def strip_wake(text: str) -> str:
    t = text
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip(" ,.?!")

# ----- Main loop -----
def main():
    ap = argparse.ArgumentParser(description="Continuous wake-word listener (Vosk small en-us 0.15)")
    ap.add_argument("--model", default=VOSK_MODEL_PATH, help="Path to vosk model dir")
    args = ap.parse_args()

    print(f"ESP32: {ESP32_BASE} -> /command")
    print("Loading Vosk model once…")
    vosk_model = Model(args.model)
    print("Ready. Say 'hey bruno ...'   (Ctrl+C to exit)")

    last_wake = 0.0

    try:
        while True:
            if not hear_wake(vosk_model):
                continue
            if time.time() - last_wake < WAKE_DEBOUNCE_S:
                continue
            last_wake = time.time()

            # Command may be in same utterance; try stripping from last wake decode:
            # (Optional: leave commented—wake handler returns only wake text typically)
            # inline_cmd = strip_wake(last_text)  # not persisted here

            cmd = hear_command(vosk_model)
            if any(w in cmd for w in WAKE_WORDS):
                cmd = strip_wake(cmd)
            if not cmd:
                cmd = "wag your tail"

            print(f"[command] {cmd}")

            plan = ask_llm(cmd)
            plan = validate_or_fix(plan)
            print("[plan]", json.dumps(plan, indent=2))

            try:
                resp = send_to_esp32(plan)
                print("[esp32]", resp)
            except Exception as e:
                print("[esp32 error]", e)

            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nbye!")

if __name__ == "__main__":
    main()
