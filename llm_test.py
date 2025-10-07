# llm_test.py
import json, sys, requests

MODEL_NAME = "phi3.5"

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
- "are you happy?" -> wag tail + turn eyes on for ~2s
- Keep it short (1-3 effects). Default duration 1500 ms.
- If unclear, default to tail wag 1.5s.
"""

def ask_llm(text: str):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":text}
            ],
            "format":"json",
            "stream": False
        },
        timeout=60
    )
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {
            "intent": "EMOTE_OR_ACTION",
            "effects": [{"part":"tail","mode":"wag","hz":6,"duration_ms":1500}]
        }

if __name__ == "__main__":
    user = " ".join(sys.argv[1:]) or "wag your tail"
    plan = ask_llm(user)
    print(json.dumps(plan, indent=2))
