# query.py
"""
Handles a user question against an already-ingested video.

    question -> top-3 chunk retrieval -> exact-timestamp frame grab
              -> vision-model answer -> (text, image_path, timestamp)

Text-to-speech is handled entirely in the browser (Web Speech API) in the
frontend that ships with app.py, so no server-side TTS plug-in is needed
unless you want a specific voice/provider later.
"""

import json
import subprocess
import tempfile
import os

import blob_storage
import search_index
from vision_model import call_vision_model


def _grab_exact_frame(video_blob_name: str, timestamp_seconds: float) -> str:
    """
    Downloads the source video (or uses a cached local copy) and pulls the
    exact frame at `timestamp_seconds` via ffmpeg. Returns a local file path.
    This is the pixel-accurate fallback -- it does NOT rely on Video
    Indexer's pre-picked keyframes.
    """
    temp_dir = tempfile.gettempdir()
    local_video_path = os.path.join(temp_dir, f"cache_{video_blob_name}")
    
    # Download ONCE per video; reuse local copy for subsequent frames
    if not os.path.exists(local_video_path):
        blob_storage.download_video_to_temp(video_blob_name, local_video_path)

    out_path = os.path.join(temp_dir, f"frame_{timestamp_seconds:.2f}.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp_seconds),
                "-i", local_video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ],
            check=True,
            capture_output=True,
        )
        return out_path
    except subprocess.CalledProcessError:
        # File has no video streams (e.g. pure audio file mp3/wav)
        return None


def _is_summary_query(question: str) -> bool:
    q = question.lower()
    keywords = [
        "explain the video", "explain me the video", "explain this video", "explain video",
        "summarize", "summary", "overview", "what is the video about", "what happens in the video",
        "walkthrough", "step by step", "total video", "entire video", "whole video"
    ]
    return any(kw in q for kw in keywords)


def _get_dynamic_step_count(chunks: list) -> int:
    if not chunks:
        return 3
    max_time = max(c.get("end_time", 0) for c in chunks)
    if max_time < 180:       # Short video (< 3 mins): 3 steps
        return 3
    elif max_time < 420:     # Medium video (3 - 7 mins): 4 steps
        return 4
    elif max_time < 900:     # Long video (7 - 15 mins): 5 steps
        return 5
    else:                    # Very long video (> 15 mins): 6 steps
        return 6


def _resolve_blob_name(chunk: dict, fallback_blob_name: str, video_map: dict) -> tuple[str, str]:
    """Returns (blob_name, display_name) for a chunk."""
    chunk_vi_id = chunk.get("video_id")
    if video_map and chunk_vi_id in video_map:
        info = video_map[chunk_vi_id]
        return info.get("blob_name", fallback_blob_name), info.get("display_name", "")
    return fallback_blob_name, ""


def answer_question(question: str, video_blob_name: str = None, video_id: str = None, video_map: dict = None) -> dict:
    structured_steps = None

    # Determine if it's a general summary query
    if _is_summary_query(question):
        # Fetch candidate chunks across video(s)
        candidate_chunks = search_index.search_top_chunks(question, video_id=video_id, video_map=video_map, top_k=6)
        if not candidate_chunks:
            return {"text": "I couldn't find anything relevant across the video library.", "snapshots": [], "structured_steps": None}

        target_count = _get_dynamic_step_count(candidate_chunks)
        chronological = sorted(candidate_chunks, key=lambda c: c["start_time"])
        
        # Select evenly spaced chunks
        if len(chronological) > target_count:
            step_size = len(chronological) / target_count
            selected_chunks = [chronological[int(i * step_size)] for i in range(target_count)]
        else:
            selected_chunks = chronological

        context_text = "\n\n".join(c["text"] for c in selected_chunks)
        snapshots = []
        frame_paths = []
        for chunk in selected_chunks:
            b_name, d_name = _resolve_blob_name(chunk, video_blob_name, video_map)
            path = _grab_exact_frame(b_name, chunk["start_time"]) if b_name else None
            frame_paths.append(path)
            snapshots.append({
                "image_path": path,
                "timestamp": chunk["start_time"],
                "video_title": d_name
            })
        
        answer_raw = call_vision_model(context_text, frame_paths, question)
        try:
            parsed = json.loads(answer_raw)
            answer_text = parsed.get("summary", "Here is the step-by-step breakdown of the video:")
            raw_steps = parsed.get("steps", [])
            structured_steps = []
            for idx, step in enumerate(raw_steps):
                snap = snapshots[idx] if idx < len(snapshots) else snapshots[-1]
                structured_steps.append({
                    "step_number": step.get("step_number", idx + 1),
                    "title": step.get("title", f"Step {idx+1}"),
                    "description": step.get("description", ""),
                    "image_path": snap["image_path"],
                    "timestamp": snap["timestamp"]
                })
        except Exception:
            answer_text = answer_raw
    else:
        # Specific query: grab only the single best frame
        top_chunks = search_index.search_top_chunks(question, video_id=video_id, video_map=video_map, top_k=3)
        if not top_chunks:
            return {"text": "I couldn't find anything relevant across the video library.", "snapshots": [], "structured_steps": None}

        context_text = "\n\n".join(c["text"] for c in top_chunks)
        best = top_chunks[0]
        b_name, d_name = _resolve_blob_name(best, video_blob_name, video_map)
        path = _grab_exact_frame(b_name, best["start_time"]) if b_name else None
        
        snapshots = [{
            "image_path": path,
            "timestamp": best["start_time"],
            "video_title": d_name
        }]
        
        answer_text = call_vision_model(context_text, [path] if path else [], question)

    # Return standard fields for backward compatibility, plus the full list of snapshots & structured steps
    return {
        "text": answer_text,
        "image_path": snapshots[0]["image_path"] if snapshots else None,
        "timestamp": snapshots[0]["timestamp"] if snapshots else None,
        "snapshots": snapshots,
        "structured_steps": structured_steps,
        "context_used": context_text,
    }