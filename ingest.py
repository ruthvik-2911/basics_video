"""
End-to-end ingestion for one video. This replaces every manual step we did
in the Video Indexer portal with code:

    upload -> wait for processing -> fetch insights -> resolve keyframe
    images -> merge transcript+OCR -> chunk -> embed -> push to search

Usage:
    python ingest.py /path/to/video.mp4 "My Video Title"
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import uuid

import blob_storage
import search_index
from video_indexer_client import VideoIndexerClient
from merge_and_chunk import build_chunks


import time

def _with_retry(fn, max_attempts: int = 3, delay: float = 2.0):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"      [Retry {attempt+1}/{max_attempts}] Transient network drop: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def ingest_video(local_path: str, display_name: str) -> tuple[str, str]:
    """Returns (vi_video_id, blob_name) so callers (e.g. app.py) can look
    the raw video back up in Blob Storage at query time."""
    video_id_local = str(uuid.uuid4())  # our own tracking id for the blob path
    blob_name = f"{video_id_local}.mp4"

    print(f"[1/6] Uploading raw video to Blob Storage...")
    _with_retry(lambda: blob_storage.upload_raw_video(local_path, blob_name))

    sas_url = blob_storage.get_blob_sas_url(blob_name)

    print(f"[2/6] Submitting video to Video Indexer via Azure Blob URL...")
    vi = VideoIndexerClient()
    vi_video_id = _with_retry(lambda: vi.upload_video(video_name=display_name, video_url=sas_url))

    print(f"[3/6] Waiting for Video Indexer to finish processing (this can take a few minutes)...")
    insights = vi.wait_for_processing(vi_video_id)

    print(f"[4/6] Building overlapping chunks from transcript + OCR...")
    chunks = build_chunks(vi_video_id, insights)
    print(f"      -> {len(chunks)} chunks built")

    print(f"[5/6] Resolving keyframe thumbnails and uploading images to Blob Storage...")
    for chunk in chunks:
        for kf in chunk.keyframes:
            image_bytes = _with_retry(lambda: vi.get_thumbnail_bytes(vi_video_id, kf.thumbnail_id))
            _with_retry(lambda: blob_storage.upload_keyframe_image(vi_video_id, kf.thumbnail_id, image_bytes))

    print(f"[6/6] Embedding chunks and pushing to Azure AI Search...")
    _with_retry(lambda: search_index.upload_chunks(chunks))

    print(f"Done. Video Indexer video_id={vi_video_id}, blob raw video name={blob_name}")
    return vi_video_id, blob_name


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <local_video_path> <display_name>")
        sys.exit(1)
    ingest_video(sys.argv[1], sys.argv[2])
