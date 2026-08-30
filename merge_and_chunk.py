# merge_and_chunk.py
"""
Turns raw Video Indexer insights JSON into search-ready chunks.

Pipeline:
  1. Merge transcript + OCR entries into one time-sorted timeline.
  2. Slide overlapping windows across that timeline (so no sentence/label
     gets cut at a boundary and silently dropped).
  3. For each window, attach every keyframe whose timestamp falls inside it
     (not just the single nearest one).
"""

from dataclasses import dataclass, field

import config


def _time_to_seconds(t: str) -> float:
    """Video Indexer times look like '0:01:14.72' -> convert to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class TimelineEntry:
    start: float
    end: float
    text: str
    kind: str  # "speech" or "ocr"


@dataclass
class KeyframeRef:
    time: float
    thumbnail_id: str


@dataclass
class Chunk:
    video_id: str
    start: float
    end: float
    text: str
    keyframes: list = field(default_factory=list)  # list[KeyframeRef]


def build_timeline(insights: dict) -> list:
    """Merge transcript + OCR entries into one time-sorted list."""
    video = insights["videos"][0]["insights"]
    entries = []

    for item in video.get("transcript", []):
        for inst in item["instances"]:
            entries.append(TimelineEntry(
                start=_time_to_seconds(inst["start"]),
                end=_time_to_seconds(inst["end"]),
                text=item["text"],
                kind="speech",
            ))

    for item in video.get("ocr", []):
        for inst in item["instances"]:
            entries.append(TimelineEntry(
                start=_time_to_seconds(inst["start"]),
                end=_time_to_seconds(inst["end"]),
                text=item["text"],
                kind="ocr",
            ))

    entries.sort(key=lambda e: e.start)
    return entries


def extract_keyframes(insights: dict) -> list:
    """Flatten every shot's keyframes into a single time-sorted list."""
    video = insights["videos"][0]["insights"]
    frames = []
    for shot in video.get("shots", []):
        for kf in shot.get("keyFrames", []):
            for inst in kf["instances"]:
                frames.append(KeyframeRef(
                    time=_time_to_seconds(inst["start"]),
                    thumbnail_id=inst["thumbnailId"],
                ))
    frames.sort(key=lambda f: f.time)
    return frames


def build_chunks(video_id: str, insights: dict) -> list:
    """
    Slide overlapping windows across the merged timeline. Each chunk's text
    combines every timeline entry whose window overlaps it (deduplicated,
    speech and OCR both included) and every keyframe whose timestamp falls
    inside the window.
    """
    timeline = build_timeline(insights)
    keyframes = extract_keyframes(insights)

    duration = _time_to_seconds(insights["videos"][0]["insights"]["duration"])

    window = config.CHUNK_WINDOW_SECONDS
    stride = config.CHUNK_STRIDE_SECONDS

    chunks = []
    t = 0.0
    while t < duration:
        window_start, window_end = t, t + window

        window_entries = [
            e for e in timeline
            if e.start < window_end and e.end > window_start
        ]
        # De-dupe identical text lines that repeat across overlapping windows' source data
        seen = set()
        lines = []
        for e in window_entries:
            key = (e.kind, e.text)
            if key not in seen:
                seen.add(key)
                prefix = "[speech]" if e.kind == "speech" else "[on-screen]"
                lines.append(f"{prefix} {e.text}")

        window_keyframes = [k for k in keyframes if window_start <= k.time < window_end]

        if lines:  # skip empty windows (e.g. pure silence with no OCR)
            chunks.append(Chunk(
                video_id=video_id,
                start=window_start,
                end=window_end,
                text="\n".join(lines),
                keyframes=window_keyframes,
            ))

        t += stride

    return chunks