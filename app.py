# app.py
"""
Thin web layer around your existing ingest_video() and answer_question().
This is the piece the readme called "still to wire up": a web/API layer
for a chatbot UI to call.

Run with:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import base64
import json
import os
import re
import requests
import tempfile
import threading
import time
import traceback
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from ingest import ingest_video
from query import answer_question

app = FastAPI(title="Video Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job + video registry.
# Fine for a single-process demo. Swap for Redis/DB if you ever run more
# than one worker process.
# ---------------------------------------------------------------------------
REGISTRY_FILE = "videos_registry.json"

def _load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_registry():
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(VIDEOS, f, indent=2)
    except Exception as e:
        print(f"Error saving registry: {e}")

# Load saved videos at app startup
VIDEOS: dict[str, dict] = _load_registry()
JOBS: dict[str, dict] = {}
# Populate JOBS dictionary for pre-existing videos so they show as "done"
for j_id, v_data in VIDEOS.items():
    JOBS[j_id] = {"status": "done", "step": "Ready"}


def _run_ingest_job(job_id: str, local_path: str, display_name: str, blob_name_hint: str):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "Uploading + indexing (this can take a few minutes)..."
    try:
        vi_video_id, blob_name = ingest_video(local_path, display_name)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "Ready"
        VIDEOS[job_id] = {
            "vi_video_id": vi_video_id,
            "blob_name": blob_name,
            "display_name": display_name,
        }
        _save_registry()
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = str(e)
        JOBS[job_id]["trace"] = traceback.format_exc()
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


@app.post("/api/upload")
async def upload_video(file: UploadFile, display_name: str = Form(...)):
    """Accepts a video file from the browser, saves it locally, and kicks
    off ingestion in a background thread so the request returns instantly."""
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    job_id = str(uuid.uuid4())
    # NOTE: ingest_video() generates its own uuid for the blob name internally
    # and prints it — we don't have it until ingestion finishes, so we surface
    # it via the /api/status endpoint once the job is done (see VIDEOS dict).
    JOBS[job_id] = {"status": "queued", "step": "Queued"}

    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, tmp_path, display_name, None),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/videos")
def get_videos():
    """Returns all previously ingested videos from the persistent registry."""
    items = []
    for j_id, v in VIDEOS.items():
        items.append({
            "job_id": j_id,
            "display_name": v.get("display_name", "Untitled Video"),
            "vi_video_id": v.get("vi_video_id"),
            "blob_name": v.get("blob_name")
        })
    return items


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    result = {"status": job["status"], "step": job.get("step")}
    if job["status"] == "done":
        result["video"] = VIDEOS[job_id]
    if job["status"] == "error":
        result["error"] = job.get("step")
    return result


@app.post("/api/ask")
async def ask(question: str = Form(...), job_id: str = Form("all")):
    video_blob_name = None
    video_id = None
    
    if job_id and job_id != "all":
        video = VIDEOS.get(job_id)
        if video:
            video_blob_name = video.get("blob_name")
            video_id = video.get("vi_video_id")
    
    video_map = {v["vi_video_id"]: {"blob_name": v["blob_name"], "display_name": v.get("display_name", "")} for v in VIDEOS.values() if "vi_video_id" in v}

    try:
        result = answer_question(
            question, 
            video_blob_name=video_blob_name, 
            video_id=video_id, 
            video_map=video_map
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"answer_question failed: {e}")

    snapshots_response = []
    for snap in result.get("snapshots", []):
        if snap.get("image_path") and os.path.exists(snap["image_path"]):
            with open(snap["image_path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            snapshots_response.append({
                "image_base64": b64,
                "timestamp": snap["timestamp"]
            })

    structured_steps_response = None
    if result.get("structured_steps"):
        structured_steps_response = []
        for step in result["structured_steps"]:
            b64 = None
            if step.get("image_path") and os.path.exists(step["image_path"]):
                with open(step["image_path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            structured_steps_response.append({
                "step_number": step.get("step_number"),
                "title": step.get("title"),
                "description": step.get("description"),
                "image_base64": b64,
                "timestamp": step.get("timestamp")
            })

    # Backward compatible fields for older/simple requests
    first_image = snapshots_response[0]["image_base64"] if snapshots_response else None
    first_timestamp = snapshots_response[0]["timestamp"] if snapshots_response else None

    return {
        "text": result["text"],
        "timestamp": first_timestamp,
        "image_base64": first_image,
        "snapshots": snapshots_response,
        "structured_steps": structured_steps_response,
    }


@app.post("/api/tts")
async def text_to_speech(text: str = Form(...), voice: str = Form("en-US-JennyNeural")):
    if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
        raise HTTPException(500, "AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not configured in .env")

    # Clean markdown syntax for clean spoken SSML
    clean_text = re.sub(r'[\*\#\_`]', '', text)
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
    clean_text = clean_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    url = f"https://{config.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
        "User-Agent": "VideoIntelligenceAssistant"
    }

    lang = "en-IN" if "en-IN" in voice else "en-US"
    ssml = f"""<speak version='1.0' xml:lang='{lang}'>
    <voice xml:lang='{lang}' xml:gender='Female' name='{voice}'>
        {clean_text}
    </voice>
</speak>"""

    try:
        resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=15)
        resp.raise_for_status()
        return Response(content=resp.content, media_type="audio/mpeg")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Azure TTS failed: {e}")


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
