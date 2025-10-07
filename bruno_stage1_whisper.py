# bruno_stage1_whisper.py
import argparse, json, queue, sys, time
import numpy as np
import sounddevice as sd
import requests
from faster_whisper import WhisperModel
from jsonschema import validate, ValidationError

# ----------------- CONFIG -----------------
WHISPER_MODEL_SIZE = "small"   # "base", "small", "medium"
USE_GPU = False                 # True if you have CUDA
SAMPLE_RATE = 16000
CHANNELS = 1
WAKE_WORDS = ("hey bruno", "hi bruno", "okay bruno")

# Recording window (soft VAD)
MAX_SECONDS = 7.0               # hard cap per utterance
MIN_SECONDS = 1.5               # minimum capture
SILENCE_HOLD = 0.6              # stop if silence lasts this long (s)
FRAME_MS = 30                   # analysis frame size
ENERGY_THRESH = 100.0           # tweak if needed (lower = more sensitive)

LANGUAGE = "en"                 # force English
# ------------------------------------------

BRUNO_SCHEMA = {
    "type": "object",
    "required": ["intent", "effects"],
    "properties": {
        "intent": {"type": "string"},
        "effects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["part", "mode"],
                "properties": {
                    "part": {
                        "type": "string",
                        "enum": [
                            "tail","head","left_ear","right_ear","left_eye","right_eye",
                            "chest","back","left_leg","right_leg"
                        ]
                    },
                    "mode": {"type": "string", "enum": ["on","off","blink","pulse","wag","chase"]},
                    "hz": {"type": "number", "minimum": 0.1, "maximum": 30},
                    "duty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "duration_ms": {"type": "integer", "minimum": 50, "maximum": 60000}
                },
                "additionalProperties": False
            },
            "minItems": 1,
            "maxItems": 5
        }
    },
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are Bruno's Skill Planner.
Map casual English into STRICT JSON controlling LEDs on an ESP32 dog robot.
Only output JSON, no extra text.

JSON schema:
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

def looks_like_wake(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in WAKE_WORDS)

def strip_wake(text: str) -> str:
    t = text.lower()
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip(" ,.?!")

def ask_llm(user_text: str):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "phi3.5",
            "messages": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_text}
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
    from copy import deepcopy
    plan = deepcopy(plan)
    try:
        validate(instance=plan, schema=BRUNO_SCHEMA)
        return plan
    except ValidationError:
        fixed = {"intent": plan.get("intent","EMOTE_OR_ACTION"), "effects": []}
        allowed_parts = {
            "tail","head","left_ear","right_ear","left_eye","right_eye","chest","back","left_leg","right_leg"
        }
        allowed_modes = {"on","off","blink","pulse","wag","chase"}
        for e in plan.get("effects", []):
            part = e.get("part","tail")
            mode = e.get("mode","wag")
            part = part if part in allowed_parts else "tail"
            mode = mode if mode in allowed_modes else "wag"
            hz = float(e.get("hz", 6))
            hz = min(max(hz, 0.1), 30)
            duty = float(e.get("duty", 0.5))
            duty = min(max(duty, 0.0), 1.0)
            dur = int(e.get("duration_ms", 1500))
            dur = min(max(dur, 50), 60000)
            fixed["effects"].append({
                "part": part, "mode": mode, "hz": hz, "duty": duty, "duration_ms": dur
            })
        if not fixed["effects"]:
            fixed = FALLBACK_PLAN
        try:
            validate(instance=fixed, schema=BRUNO_SCHEMA)
            return fixed
        except ValidationError:
            return FALLBACK_PLAN

# ---- Simple energy-gated recording (no C++ VAD needed) ----
def record_utterance():
    """
    Capture audio until we detect sustained silence or hit MAX_SECONDS.
    Returns float32 PCM in [-1, 1].
    """
    frame_len = int(SAMPLE_RATE * (FRAME_MS/1000.0))
    silence_frames_needed = int(SILENCE_HOLD / (FRAME_MS/1000.0))
    buf = []
    silent_run = 0
    total_frames = 0
    start = time.time()
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata.copy())

    print("🎙️  listening… say 'hey bruno ...'")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=cb, blocksize=frame_len):
        while True:
            try:
                chunk = q.get(timeout=MAX_SECONDS)
