import json, time, queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import requests
from jsonschema import validate, ValidationError

# ---- CONFIG ----
SAMPLE_RATE = 16000
MODEL_PATH = r".\models\vosk-model-en-us-0.22"   
WAKE_WORDS = ("hey bruno","hi bruno","okay bruno", "bruno")
LISTEN_WAKE_S = 2.5
LISTEN_CMD_S  = 4.0
# ---------------

SYSTEM_PROMPT = """You are Bruno's Skill Planner.
Only output STRICT JSON with:
{"intent":"EMOTE_OR_ACTION","effects":[{"part":"tail|head|left_ear|right_ear|left_eye|right_eye|chest|back|left_leg|right_leg","mode":"on|off|blink|pulse|wag|chase","hz":opt,"duty":opt,"duration_ms":opt}]}
Defaults: hz=6, duty=0.5, duration_ms=1500. Keep 1–3 effects.
"""

FALLBACK_PLAN = {"intent":"EMOTE_OR_ACTION","effects":[{"part":"tail","mode":"wag","hz":6,"duration_ms":1500}]}

BRUNO_SCHEMA = {
  "type":"object","required":["intent","effects"],
  "properties":{
    "intent":{"type":"string"},
    "effects":{"type":"array","minItems":1,"maxItems":5,"items":{
      "type":"object","required":["part","mode"],
      "properties":{
        "part":{"type":"string","enum":["tail","head","left_ear","right_ear","left_eye","right_eye","chest","back","left_leg","right_leg"]},
        "mode":{"type":"string","enum":["on","off","blink","pulse","wag","chase"]},
        "hz":{"type":"number","minimum":0.1,"maximum":30},
        "duty":{"type":"number","minimum":0.0,"maximum":1.0},
        "duration_ms":{"type":"integer","minimum":50,"maximum":60000}
      },
      "additionalProperties": False
    }}
  },
  "additionalProperties": False
}

def ask_llm(user_text:str):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model":"phi3.5",
        "messages":[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_text}
        ],
        "format":"json","stream":False,"options":{"temperature":0}
    }, timeout=60)
    r.raise_for_status()
    content = r.json().get("message",{}).get("content","").strip()
    try: return json.loads(content)
    except: return FALLBACK_PLAN

def validate_or_fix(plan:dict)->dict:
    try:
        validate(instance=plan, schema=BRUNO_SCHEMA)
        return plan
    except:
        fixed={"intent":plan.get("intent","EMOTE_OR_ACTION"),"effects":[]}
        allowed_parts={"tail","head","left_ear","right_ear","left_eye","right_eye","chest","back","left_leg","right_leg"}
        allowed_modes={"on","off","blink","pulse","wag","chase"}
        for e in plan.get("effects",[]):
            part = e.get("part","tail");         part = part if part in allowed_parts else "tail"
            mode = e.get("mode","wag");          mode = mode if mode in allowed_modes else "wag"
            hz   = float(e.get("hz",6));         hz   = min(max(hz,0.1),30)
            duty = float(e.get("duty",0.5));     duty = min(max(duty,0.0),1.0)
            dur  = int(e.get("duration_ms",1500)); dur = min(max(dur,50),60000)
            fixed["effects"].append({"part":part,"mode":mode,"hz":hz,"duty":duty,"duration_ms":dur})
        return fixed or FALLBACK_PLAN

def _rec_stream(rec, seconds:float):
    q = queue.Queue()
    def cb(indata, frames, t, status):
        q.put(bytes(indata))
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16', channels=1, callback=cb):
        start = time.time()
        while time.time() - start < seconds:
            data = q.get()
            rec.AcceptWaveform(data)
        try:
            return json.loads(rec.FinalResult()).get("text","").lower().strip()
        except:
            return ""

def hear_wake(vosk_model) -> bool:
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetGrammar('["hey bruno","hi bruno","okay bruno","[unk]"]')
    rec.SetWords(True)
    print("🎙️  listening for wake word…")
    text = _rec_stream(rec, LISTEN_WAKE_S)
    print("[wake]:", text or "(silence)")
    return any(w in text for w in WAKE_WORDS)

def hear_command(vosk_model) -> str:
    # light grammar to bias decoding; comment these two lines out if it hurts you
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)
    # optional: rec.SetGrammar(json.dumps(["wag your tail","blink your eyes","are you happy","tail","head","blink","wag","on","off","eyes","ears","left","right","fast","slow","chest","back","leg"] + ["[unk]"]))
    print("🎙️  say the command…")
    text = _rec_stream(rec, LISTEN_CMD_S)
    print("[cmd]:", text or "(silence)")
    return text

def strip_wake(text:str)->str:
    t = text
    for w in WAKE_WORDS:
        t = t.replace(w, "")
    return t.strip(" ,.?!")

def main():
    print("Loading Vosk model once…")
    model = Model(MODEL_PATH)  # <- this is the expensive step; done only once
    print("Ready. Say 'hey bruno ...' anytime. Ctrl+C to exit.")
    while True:
        if not hear_wake(model):
            continue
        cmd = hear_command(model)
        if not cmd:
            print("didn't catch that, try again.")
            continue
        if any(w in cmd for w in WAKE_WORDS):
            cmd = strip_wake(cmd) or "wag your tail"
        print("[final command]:", cmd)
        plan = ask_llm(cmd)
        safe_plan = validate_or_fix(plan)
        print(json.dumps(safe_plan, indent=2))

if __name__ == "__main__":
    main()
