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


def _clean_answer_text(raw_text: str) -> str:
    if not raw_text or not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()

    # Strip code block wrappers like ```json ... ``` or ```markdown ... ```
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1 and text.endswith("```"):
            text = text[first_newline + 1:-3].strip()

    # If the response was wrapped in a JSON object, safely extract its primary content
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ["answer", "summary", "response", "text", "description"]:
                    if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                        return parsed[key].strip()
                if "steps" in parsed and isinstance(parsed["steps"], list):
                    parts = []
                    if "summary" in parsed:
                        parts.append(str(parsed["summary"]))
                    for s in parsed["steps"]:
                        if isinstance(s, dict):
                            stitle = s.get("title", "")
                            sdesc = s.get("description", "")
                            parts.append(f"**{stitle}**\n{sdesc}")
                    return "\n\n".join(parts).strip()
        except Exception:
            pass

    return text


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


def format_citation_badge(file_name: str, source_type: str, location: str) -> str:
    s_type = (source_type or "document").lower()
    if s_type == "video":
        return f"🎬 Video: {file_name} [{location}]"
    elif s_type == "audio":
        return f"🎵 Audio: {file_name} [{location}]"
    elif s_type == "xlsx":
        return f"📊 Spreadsheet: {file_name} ({location})"
    elif s_type == "image":
        return f"🖼️ Image: {file_name}"
    else:
        return f"📄 Document: {file_name} ({location})"


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


def answer_question(question: str, video_blob_name: str = None, video_id: str = None, video_map: dict = None, language: str = "English") -> dict:
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
            
            # Format clean location string
            final_loc = loc
            if not final_loc:
                if s_type == "video":
                    mins = int(chunk.get('start_time', 0) // 60)
                    secs = int(chunk.get('start_time', 0) % 60)
                    final_loc = f"{mins:02d}:{secs:02d}"
                else:
                    final_loc = f"Page {int(chunk.get('start_time', 1))}"

            citation_item = {
                "file_name": d_name or b_name or "Document",
                "source_type": s_type,
                "location": final_loc,
                "citation_badge": format_citation_badge(d_name or b_name or "Document", s_type, final_loc),
                "image_path": path,
            }
            citations.append(citation_item)
            snapshots.append({
                "image_path": path,
                "timestamp": chunk.get("start_time", 0),
                "video_title": d_name or b_name or "File",
                "location": final_loc,
                "source_type": s_type,
                "citation_badge": citation_item["citation_badge"],
            })
        
        answer_raw = call_vision_model(context_text, frame_paths, question, language=language)
        try:
            parsed = json.loads(answer_raw)
            answer_text = parsed.get("summary", "Here is the step-by-step breakdown:")
            raw_steps = parsed.get("steps", [])
            structured_steps = []
            for idx, step in enumerate(raw_steps):
                snap = snapshots[idx] if idx < len(snapshots) else snapshots[-1]
                s_badge = snap.get("citation_badge") or format_citation_badge(snap.get("video_title", "File"), snap.get("source_type", "video"), snap.get("location", ""))
                structured_steps.append({
                    "step_number": step.get("step_number", idx + 1),
                    "title": step.get("title", f"Step {idx+1}"),
                    "description": step.get("description", ""),
                    "image_path": snap["image_path"],
                    "timestamp": snap["timestamp"],
                    "source_type": snap.get("source_type", "video"),
                    "location": snap.get("location", ""),
                    "file_name": snap.get("video_title", ""),
                    "citation_badge": s_badge,
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
        
        final_best_loc = loc
        if not final_best_loc:
            if s_type == "video":
                mins = int(best.get('start_time', 0) // 60)
                secs = int(best.get('start_time', 0) % 60)
                final_best_loc = f"{mins:02d}:{secs:02d}"
            else:
                final_best_loc = f"Page {int(best.get('start_time', 1))}"

        citations = []
        for chunk in top_chunks:
            cb_name, cd_name, cs_type, cloc = _resolve_blob_info(chunk, video_blob_name, video_map)
            cpath = _get_image_for_chunk(cb_name, cs_type, chunk.get("start_time", 0))
            
            c_loc = cloc
            if not c_loc:
                if cs_type == "video":
                    mins = int(chunk.get('start_time', 0) // 60)
                    secs = int(chunk.get('start_time', 0) % 60)
                    c_loc = f"{mins:02d}:{secs:02d}"
                else:
                    c_loc = f"Page {int(chunk.get('start_time', 1))}"

            citations.append({
                "file_name": cd_name or cb_name or "Document",
                "source_type": cs_type,
                "location": c_loc,
                "citation_badge": format_citation_badge(cd_name or cb_name or "Document", cs_type, c_loc),
                "image_path": cpath,
            })

        best_badge = format_citation_badge(d_name or b_name or "File", s_type, final_best_loc)
        snapshots = [{
            "image_path": path,
            "timestamp": best.get("start_time", 0),
            "video_title": d_name or b_name or "File",
            "location": final_best_loc,
            "source_type": s_type,
            "citation_badge": best_badge,
        }]
        
        raw_answer = call_vision_model(context_text, [path] if path else [], question, language=language)
        answer_text = _clean_answer_text(raw_answer)

    # Return standard fields for backward compatibility, plus citations & full structured steps
    return {
        "text": answer_text,
        "image_path": snapshots[0]["image_path"] if snapshots else None,
        "timestamp": snapshots[0]["timestamp"] if snapshots else None,
        "location": snapshots[0].get("location") if snapshots else "",
        "file_name": snapshots[0].get("video_title") if snapshots else "",
        "snapshots": snapshots,
        "citations": citations,
        "structured_steps": structured_steps,
        "context_used": context_text,
    }