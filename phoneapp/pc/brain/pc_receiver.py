"""
Bruno PC Receiver / Hub (WebSocket + WebRTC)

- Listens on ws://0.0.0.0:8765
- Handles WebRTC signaling from the Android app (offer -> answer).
- Receives the phone's front-camera stream (video track).
- Periodically sends JSON control packets to the phone to:
    * update face (eyes/mouth/blink)
    * speak via TTS on the phone (tts.text)
- Designed so you can later plug in OpenCV/emotion detection.

Requires:
  pip install aiortc websockets

Run:
  python pc_receiver.py
"""

import asyncio
import json
import time
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole  # placeholder consumer

# ---- Config ----
HOST = "0.0.0.0"
PORT = 8765

# We may support multiple 'faces' later; for now there will be one Android phone.
FACE_CLIENTS: Set[WebSocketServerProtocol] = set()


async def broadcast_face(payload: dict) -> None:
    """Send a JSON message to all connected face clients (Android app)."""
    if not FACE_CLIENTS:
        return
    msg = json.dumps(payload)
    await asyncio.gather(
        *[ws.send(msg) for ws in list(FACE_CLIENTS) if not ws.closed],
        return_exceptions=True,
    )


async def handle_android_face(ws: WebSocketServerProtocol) -> None:
    """
    Handle messages from the Android face app:
      - role announcement: {"role":"face"}
      - WebRTC offer: {"type":"offer","sdp":"..."}
      - optional ICE candidates: {"type":"ice","candidate":{...}}
    Responds with:
      - WebRTC answer: {"type":"answer","sdp":"..."}
      - periodic control packets: {"target":"face", ...}
    """
    # Create a new PeerConnection for this client
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        print(f"[PC] Track received: kind={track.kind}")
        # For now we "blackhole" the media (consume and drop).
        # Later you'll plug OpenCV/AI here to process frames.
        if track.kind == "video":
            bh = MediaBlackhole()
            # MediaBlackhole has a "recv" coroutine; attach it to this track
            # by creating a task that pulls frames behind the scenes.
            # (Alternatively use aiortc MediaRecorder to write to file.)
            # Note: This is a lightweight placeholder.
            asyncio.create_task(bh.recv())
        elif track.kind == "audio":
            # No action for audio in v0.3; could be used for voice activity later.
            pass

    # Simple message loop
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            # 1) First message from Android is commonly {"role":"face"}
            if msg.get("role") == "face":
                FACE_CLIENTS.add(ws)
                print("[WS] Android face connected")
                continue

            # 2) WebRTC offer -> create answer
            if msg.get("type") == "offer":
                offer = RTCSessionDescription(sdp=msg["sdp"], type="offer")
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
                print("[RTC] Sent answer")
                continue

            # 3) ICE from Android (not strictly needed server-side here, ok to ignore)
            if msg.get("type") == "ice":
                # If you later add trickle ICE exchange, parse and add remote candidates here.
                continue

            # (Any other client-to-server messages can be handled here.)
    finally:
        # Cleanup on disconnect
        if ws in FACE_CLIENTS:
            FACE_CLIENTS.discard(ws)
            print("[WS] Android face disconnected")
        await pc.close()


async def control_loop() -> None:
    """
    Demo loop: every 4 seconds send a face+TTS packet to the phone.
    Replace or augment this with your real orchestrator (STT/LLM/vision).
    """
    while True:
        payload = {
            "target": "face",
            "ts": time.time(),
            "expression": "eyes_wide",
            "mouth": "smile",
            "blink": {"mode": "auto", "period_ms": 2400},
            "tts": {"text": "Hi! I am Bruno."},
        }
        await broadcast_face(payload)
        await asyncio.sleep(4)


async def ws_router(ws: WebSocketServerProtocol):
    """
    Single entrypoint for new WebSocket connections.
    We keep it simple: every client is expected to be the Android face for v0.3.
    """
    await handle_android_face(ws)


async def main():
    print(f"[WS] Listening on ws://{HOST}:{PORT}")
    async with websockets.serve(ws_router, HOST, PORT):
        # Run the demo control loop forever
        await control_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down")
