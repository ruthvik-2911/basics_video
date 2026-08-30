# Video Chatbot — Ingestion & Query Pipeline

This is the automated version of everything we did manually in the Azure
Portal and Video Indexer web UI. Once this is running, uploading a video and
asking a question needs zero manual clicking.

## What each file does

| File | Purpose |
|---|---|
| `config.py` | Reads all settings from environment variables (see `.env.example`) |
| `video_indexer_client.py` | Talks to the Video Indexer API: uploads video, polls for completion, fetches insights JSON, fetches keyframe images |
| `blob_storage.py` | Uploads raw videos and keyframe images to your `videochatbotstore1` storage account |
| `merge_and_chunk.py` | Merges transcript + OCR into one timeline, builds overlapping 15s chunks, links each chunk to its nearby keyframes |
| `search_index.py` | Creates the Azure AI Search index, embeds chunk text locally (no Azure OpenAI needed), uploads/searches |
| `ingest.py` | Orchestrates the full ingestion pipeline for one video — **run this when a video is uploaded** |
| `query.py` | Orchestrates answering one question — retrieval, exact-timestamp frame grab, leaves a slot for your vision model call |

## Setup

1. `pip install -r requirements.txt`
2. `ffmpeg` must be installed and on your PATH (used for the exact-timestamp frame grab).
3. Copy `.env.example` to `.env` and fill in every value:
   - `VI_*` values come from your `video-chatbot-indexer29` resource's Overview page in the Azure Portal (Account ID, Subscription ID, etc.)
   - `BLOB_CONNECTION_STRING` comes from your `videochatbotstore1` storage account → Access keys
   - `SEARCH_ENDPOINT` / `SEARCH_ADMIN_KEY` come from an Azure AI Search resource (create one — F0 free tier is fine to start; search "Azure AI Search" in the Portal same way we created the other resources)
4. Load the `.env` file before running (e.g. `python-dotenv`, or `export $(cat .env | xargs)` on Linux/Mac).
5. Authenticate for Azure AD: run `az login` once locally, or set up a managed identity/service principal in production. This is what `DefaultAzureCredential` in `video_indexer_client.py` uses.

## Running ingestion

```bash
python ingest.py /path/to/your/video.mp4 "Hotel Booking Walkthrough"
```

This runs the exact 6 steps we validated manually in the portal, just automated:
upload → index → wait → chunk → resolve thumbnails → push to search.

## Running a query

```python
from query import answer_question

result = answer_question(
    "what do I do at the payment step?",
    video_blob_name="<the blob_name printed by ingest.py>",
)
print(result["text"])
print(result["image_path"])   # exact-timestamp frame, ready to show in the UI
print(result["timestamp"])
```

## Still to wire up

- `query.py` has two placeholder comments: the vision-model call (send
  `context_text` + `frame_path` + the question to a vision-capable model)
  and text-to-speech (send the returned answer text to Azure AI Speech or
  another TTS provider). These were left as plug-in points since the choice
  of provider is yours to make.
- A thin web/API layer around `ingest_video()` and `answer_question()` for
  your actual chatbot UI to call.