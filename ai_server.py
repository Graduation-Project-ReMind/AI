"""
ai_server.py — ReMind AI Eye-Tracking Server (FastAPI · WebSocket)
==================================================================
بيركب على الـ "New" real-time architecture بتاعكم:

  1) الفرونت يهيت Laravel /activities/{id}/sessions (أو /tests/.../sessions)
  2) Laravel يعمل Session + websocket_token، وينده FastAPI POST /sessions
     عشان يسجّل التوكن (Laravel = control plane)
  3) Laravel يرجّع للفرونت الـ ws_url + token + session_id
  4) الفرونت يفتح WebSocket مباشرة هنا ويبعت الفريمات لايف (binary JPEG)
  5) السيرفر يحلل كل فريم لحظياً ويرجّع النتيجة لايف
  6) عند الإيقاف/قطع الاتصال → السيرفر يبني الـ report ويبعته webhook لـ Laravel

نفس الـ flow للـ Test module والـ Activity module — الفرق بس في
أي Laravel controller بيعمل الـ session وأي report model بيتخزّن.

WS protocol (ws://HOST/ws/eye-tracking?session_id=..&token=..):
  client → server:  binary JPEG frames  |  {"type":"stop"}
  server → client:  {"type":"ready"} | {"type":"result",..} | {"type":"report",..} | {"type":"error",..}
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gaze_tracker import GazeTracker

# ─── Config ───────────────────────────────────────────────────────────────────
LARAVEL_WEBHOOK_URL   = os.getenv("LARAVEL_WEBHOOK_URL",   "http://localhost:8000/api/ai/activity-webhook")
LARAVEL_WEBHOOK_TOKEN = os.getenv("LARAVEL_WEBHOOK_TOKEN", "your-secret-webhook-token")
# توكن مشترك بين Laravel و FastAPI لتأمين الـ /sessions register call
SERVICE_TOKEN         = os.getenv("AI_SERVICE_TOKEN", "shared-service-token")
# الـ URL اللي الفرونت هيتصل بيه (FastAPI مباشرة أو عبر Nginx تحت دومين Laravel)
PUBLIC_WS_URL         = os.getenv("PUBLIC_WS_URL", "ws://localhost:8001")
HOST                  = os.getenv("AI_HOST", "0.0.0.0")
PORT                  = int(os.getenv("AI_PORT", 8001))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ReMind AI Eye-Tracking")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Session Store ────────────────────────────────────────────────────────────
# session_id → { tracker, token, subject_name, module, connected, started_at }
_sessions: dict = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════════════════════════════════
#  Models
# ════════════════════════════════════════════════════════════════════════════
class RegisterRequest(BaseModel):
    session_id: str
    websocket_token: str
    subject_name: str = "Subject"
    module: str = "activity"          # 'activity' | 'test'


# ════════════════════════════════════════════════════════════════════════════
#  Report building + webhook
# ════════════════════════════════════════════════════════════════════════════
def _build_report(session_id: str) -> dict:
    sess = _sessions.get(session_id)
    if not sess:
        return {}

    m = sess["tracker"].metrics
    total_time = m.total_fixation_duration + m.total_distraction_duration

    return {
        "session_id":             session_id,
        "module":                 sess["module"],
        "subject_name":           sess["subject_name"],
        "total_frames":           m.total_frames,
        "no_face_frames":         0, # no_face_frames not tracked by metrics
        "session_duration":       round(total_time, 2),
        "attention_percentage":   round(m.attention_percentage, 1),
        "distraction_percentage": round(100.0 - m.attention_percentage, 1),

        # متابعة النظر للموبايل
        "on_screen_percentage":   round(m.attention_percentage, 1),
        "off_screen_percentage":  round(100.0 - m.attention_percentage, 1),
        "on_screen_seconds":      round(m.total_fixation_duration, 2),
        "off_screen_seconds":     round(m.total_distraction_duration, 2),

        "blink_count":            m.blink_count,
        "blink_rate_per_min":     round(m.blink_rate_per_min, 1),
        "fixation_count":         m.fixation_count,
        "avg_fixation_duration":  round(m.avg_fixation_duration, 2),
        "saccade_count":          m.saccade_count,
        "saccade_rate":           round(m.saccade_rate, 3),
        "gaze_stability":         round(max(0.0, 100.0 - (m.saccade_rate * 10)), 1),
        "gaze_directions":        dict(m.direction_counts),
        "timestamp":              _utcnow(),
    }


def _send_webhook_sync(report: dict):
    try:
        res = requests.post(
            LARAVEL_WEBHOOK_URL,
            json=report,
            headers={
                "X-AI-Webhook-Token": LARAVEL_WEBHOOK_TOKEN,
                "Content-Type":       "application/json",
                "Accept":             "application/json",
            },
            timeout=15,
        )
        logger.info(f"Webhook sent for session {report.get('session_id')} → {res.status_code}")
    except Exception as e:
        logger.error(f"Webhook failed for session {report.get('session_id')}: {e}")


async def _finalize_session(session_id: str) -> dict:
    """يبني الـ report، ينضّف الجلسة، ويبعت الـ webhook (في الخلفية)."""
    if session_id not in _sessions:
        return {}
    report = _build_report(session_id)
    _sessions.pop(session_id, None)
    logger.info(f"Session finalized: {session_id} — {report.get('total_frames')} frames, "
                f"attention={report.get('attention_percentage')}%, "
                f"on-screen={report.get('on_screen_percentage')}%")
    asyncio.create_task(asyncio.to_thread(_send_webhook_sync, report))
    return report


def _analyze_frame_sync(tracker: GazeTracker, jpeg_bytes: bytes) -> dict:
    """Decode + process (بيتنده داخل thread عشان ما يبلّكش الـ event loop)."""
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"type": "error", "error": "invalid_image"}

    _, gf = tracker.process_frame(frame)
    m = tracker.metrics

    if gf is None:
        return {
            "type": "result", "face_found": False, "timestamp": _utcnow(),
            "metrics": {
                "attention_percentage": round(m.attention_percentage, 1),
                "on_screen_percentage": round(m.attention_percentage, 1),
                "blink_count": m.blink_count,
                "total_frames": m.total_frames,
            },
        }

    return {
        "type": "result", "face_found": True, "timestamp": _utcnow(),
        "gaze": {
            "direction":         gf.gaze_direction,
            "blink":             gf.blink_detected,
            "eyes_closed":       gf.blink_detected,
            "looking_at_screen": gf.gaze_direction == "center",
            "attention_score":   round(gf.attention_score, 1),
            "left_ratio":        round(gf.left_ratio, 3),
            "right_ratio":       round(gf.right_ratio, 3),
            "vertical_ratio":    round(gf.vertical_ratio, 3),
            "ear":               round(gf.ear, 3),
            "head_yaw":          0.0,
            "head_pitch":        0.0,
            "head_roll":         0.0,
        },
        "metrics": {
            "attention_percentage": round(m.attention_percentage, 1),
            "on_screen_percentage": round(m.attention_percentage, 1),
            "blink_count":          m.blink_count,
            "fixation_count":       m.fixation_count,
            "saccade_count":        m.saccade_count,
            "total_frames":         m.total_frames,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
#  HTTP — control plane (بيتنده من Laravel)
# ════════════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    active = sum(1 for s in _sessions.values() if s["connected"])
    return {"status": "ok", "active_sessions": active,
            "registered_sessions": len(_sessions), "timestamp": _utcnow()}


@app.post("/sessions")
async def register_session(req: RegisterRequest, x_service_token: str = Header(None)):
    """
    Laravel بينده هنا وقت الـ start عشان يسجّل الـ session + التوكن قبل ما
    الفرونت يتصل بالـ WebSocket. (Laravel = control plane).
    """
    if x_service_token != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    _sessions[req.session_id] = {
        "tracker":      GazeTracker(min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7),
        "token":        req.websocket_token,
        "subject_name": req.subject_name,
        "module":       req.module,
        "connected":    False,
        "started_at":   _utcnow(),
    }
    logger.info(f"Session registered: {req.session_id} (module={req.module})")
    return {
        "success":    True,
        "session_id": req.session_id,
        "ws_url":     f"{PUBLIC_WS_URL}/ws/eye-tracking",
        "timestamp":  _utcnow(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  WebSocket — data plane (الفرونت بيتصل مباشرة + live streaming)
# ════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/eye-tracking")
async def ws_eye_tracking(ws: WebSocket):
    session_id = ws.query_params.get("session_id", "")
    token      = ws.query_params.get("token", "")

    await ws.accept()

    sess = _sessions.get(session_id)
    if not sess or sess["token"] != token:
        await ws.send_json({"type": "error", "error": "invalid_session_or_token"})
        await ws.close(code=4003)
        logger.warning(f"WS rejected (bad token/session): {session_id}")
        return

    sess["connected"] = True
    tracker = sess["tracker"]
    logger.info(f"WS connected: {session_id} (module={sess['module']})")
    await ws.send_json({"type": "ready", "session_id": session_id})

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # فريم binary (المسار الأساسي — أخف وأسرع)
            data = msg.get("bytes")
            if data is not None:
                result = await asyncio.to_thread(_analyze_frame_sync, tracker, data)
                await ws.send_json(result)
                continue

            # رسائل نصية: control أو فريم base64 (fallback)
            text = msg.get("text")
            if text is not None:
                try:
                    ctrl = json.loads(text)
                except Exception:
                    continue
                mtype = ctrl.get("type")
                if mtype == "stop":
                    break
                if mtype == "frame" and ctrl.get("data"):
                    import base64
                    raw = base64.b64decode(ctrl["data"])
                    result = await asyncio.to_thread(_analyze_frame_sync, tracker, raw)
                    await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WS error on session {session_id}: {e}")
    finally:
        report = await _finalize_session(session_id)
        try:
            await ws.send_json({"type": "report", "session_id": session_id, "report": report})
            await ws.close()
        except Exception:
            pass


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  ReMind — AI Eye-Tracking Server (FastAPI · WebSocket)")
    print(f"  http://{HOST}:{PORT}")
    print(f"  WS  → {PUBLIC_WS_URL}/ws/eye-tracking")
    print(f"  Register (Laravel) → POST /sessions")
    print(f"  Webhook → {LARAVEL_WEBHOOK_URL}")
    print("=" * 60)
    uvicorn.run("ai_server:app", host=HOST, port=PORT, ws="websockets")
