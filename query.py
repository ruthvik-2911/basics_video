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


def _resolve_blob_info(chunk: dict, fallback_blob_name: str, video_map: dict) -> tuple[str, str, str, str]:
    """Returns (blob_name, display_name, source_type, location) for a chunk."""
    chunk_id = chunk.get("video_id")
    s_type = chunk.get("source_type", "video")
    
    # Extract location from keyframe_thumbnail_ids if available (e.g. "loc:Page 2")
    loc = ""
    for thumb in chunk.get("keyframe_thumbnail_ids", []) or []:
        if thumb and thumb.startswith("loc:"):
            loc = thumb[4:]
            break

    if video_map and chunk_id in video_map:
        info = video_map[chunk_id]
        blob = info.get("blob_name", fallback_blob_name)
        disp = info.get("display_name", "")
        s_type = info.get("source_type", s_type)
        return blob, disp, s_type, loc

    return fallback_blob_name, "", s_type, loc


def _get_image_for_chunk(blob_name: str, source_type: str, timestamp_or_page: float) -> str:
    """Returns local image path for a video frame or standalone image, or None for documents."""
    if not blob_name:
        return None

    temp_dir = tempfile.gettempdir()

    # If it's a standalone image file
    if source_type == "image":
        local_img_path = os.path.join(temp_dir, f"cache_{blob_name}")
        if not os.path.exists(local_img_path):
            try:
                blob_storage.download_video_to_temp(blob_name, local_img_path)
            except Exception:
                return None
        return local_img_path

    # If it's a video file, extract exact frame
    if source_type == "video":
        return _grab_exact_frame(blob_name, timestamp_or_page)

    return None


def answer_question(question: str, video_blob_name: str = None, video_id: str = None, video_map: dict = None) -> dict:
    structured_steps = None

    # Determine if it's a general summary query
    if _is_summary_query(question):
        # Fetch candidate chunks across library
        candidate_chunks = search_index.search_top_chunks(question, video_id=video_id, video_map=video_map, top_k=6)
        if not candidate_chunks:
            return {"text": "I couldn't find anything relevant across your knowledge library.", "snapshots": [], "citations": [], "structured_steps": None}

        target_count = _get_dynamic_step_count(candidate_chunks)
        chronological = sorted(candidate_chunks, key=lambda c: c.get("start_time", 0))
        
        # Select evenly spaced chunks
        if len(chronological) > target_count:
            step_size = len(chronological) / target_count
            selected_chunks = [chronological[int(i * step_size)] for i in range(target_count)]
        else:
            selected_chunks = chronological

        context_text = "\n\n".join(c["text"] for c in selected_chunks)
        snapshots = []
        citations = []
        frame_paths = []

        for chunk in selected_chunks:
            b_name, d_name, s_type, loc = _resolve_blob_info(chunk, video_blob_name, video_map)
            path = _get_image_for_chunk(b_name, s_type, chunk.get("start_time", 0))
            if path:
                frame_paths.append(path)
            
            # Record citation
            citation_item = {
                "file_name": d_name or b_name or "Document",
                "source_type": s_type,
                "location": loc or (f"{chunk.get('start_time', 0):.2f}s" if s_type == "video" else f"Page {int(chunk.get('start_time', 1))}"),
                "image_path": path,
            }
            citations.append(citation_item)
            snapshots.append({
                "image_path": path,
                "timestamp": chunk.get("start_time", 0),
                "video_title": d_name,
                "location": loc or "",
                "source_type": s_type
            })
        
        answer_raw = call_vision_model(context_text, frame_paths, question)
        try:
            parsed = json.loads(answer_raw)
            answer_text = parsed.get("summary", "Here is the step-by-step breakdown:")
            raw_steps = parsed.get("steps", [])
            structured_steps = []
            for idx, step in enumerate(raw_steps):
                snap = snapshots[idx] if idx < len(snapshots) else snapshots[-1]
                structured_steps.append({
                    "step_number": step.get("step_number", idx + 1),
                    "title": step.get("title", f"Step {idx+1}"),
                    "description": step.get("description", ""),
                    "image_path": snap["image_path"],
                    "timestamp": snap["timestamp"],
                    "source_type": snap.get("source_type", "video"),
                    "location": snap.get("location", "")
                })
        except Exception:
            answer_text = answer_raw
    else:
        # Specific query: grab top chunks
        top_chunks = search_index.search_top_chunks(question, video_id=video_id, video_map=video_map, top_k=3)
        if not top_chunks:
            return {"text": "I couldn't find anything relevant across your knowledge library.", "snapshots": [], "citations": [], "structured_steps": None}

        context_text = "\n\n".join(c["text"] for c in top_chunks)
        best = top_chunks[0]
        b_name, d_name, s_type, loc = _resolve_blob_info(best, video_blob_name, video_map)
        path = _get_image_for_chunk(b_name, s_type, best.get("start_time", 0))
        
        citations = []
        for chunk in top_chunks:
            cb_name, cd_name, cs_type, cloc = _resolve_blob_info(chunk, video_blob_name, video_map)
            cpath = _get_image_for_chunk(cb_name, cs_type, chunk.get("start_time", 0))
            citations.append({
                "file_name": cd_name or cb_name or "Document",
                "source_type": cs_type,
                "location": cloc or (f"{chunk.get('start_time', 0):.2f}s" if cs_type == "video" else f"Page {int(chunk.get('start_time', 1))}"),
                "image_path": cpath,
            })

        snapshots = [{
            "image_path": path,
            "timestamp": best.get("start_time", 0),
            "video_title": d_name,
            "location": loc or "",
            "source_type": s_type
        }]
        
        answer_text = call_vision_model(context_text, [path] if path else [], question)

    # Return standard fields for backward compatibility, plus citations & full structured steps
    return {
        "text": answer_text,
        "image_path": snapshots[0]["image_path"] if snapshots else None,
        "timestamp": snapshots[0]["timestamp"] if snapshots else None,
        "snapshots": snapshots,
        "citations": citations,
        "structured_steps": structured_steps,
        "context_used": context_text,
    }