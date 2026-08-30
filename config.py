# config.py
"""
Central configuration. All values come from environment variables so no
secrets live in code. Copy `.env.example` to `.env` and fill in your values,
then load it (e.g. with python-dotenv) before running anything.
"""

import os

# --- Video Indexer (ARM account) ---
VI_SUBSCRIPTION_ID = os.environ["VI_SUBSCRIPTION_ID"]        # Azure subscription ID
VI_RESOURCE_GROUP = os.environ["VI_RESOURCE_GROUP"]          # e.g. "rg-video-chatbot"
VI_ACCOUNT_NAME = os.environ["VI_ACCOUNT_NAME"]               # e.g. "video-chatbot-indexer29"
VI_ACCOUNT_ID = os.environ["VI_ACCOUNT_ID"]                   # the "Account ID" shown on the resource Overview page
VI_LOCATION = os.environ.get("VI_LOCATION", "eastus")         # ARM region slug (lowercase, no spaces)

# --- Azure Blob Storage ---
BLOB_CONNECTION_STRING = os.environ["BLOB_CONNECTION_STRING"]
BLOB_CONTAINER_RAW_VIDEOS = os.environ.get("BLOB_CONTAINER_RAW_VIDEOS", "raw-videos")
BLOB_CONTAINER_KEYFRAMES = os.environ.get("BLOB_CONTAINER_KEYFRAMES", "key-frames")

# --- Azure AI Search ---
SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]                # https://<name>.search.windows.net
SEARCH_ADMIN_KEY = os.environ["SEARCH_ADMIN_KEY"]
SEARCH_INDEX_NAME = os.environ.get("SEARCH_INDEX_NAME", "video-chunks")

# --- Chunking ---
CHUNK_WINDOW_SECONDS = 15
CHUNK_STRIDE_SECONDS = 7.5   # overlap = window - stride

# --- Embedding model (local, free, no Azure OpenAI dependency) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, runs locally via sentence-transformers
EMBEDDING_DIM = 384

# --- Azure Speech (Neural Voice) ---
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")