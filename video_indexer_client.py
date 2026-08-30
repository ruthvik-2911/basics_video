# video_indexer_client.py
"""
Wraps the Azure AI Video Indexer data-plane API for an ARM-based account.

Auth flow (ARM accounts only, this is different from trial accounts):
  1. Get an ARM bearer token via Azure AD (DefaultAzureCredential handles
     `az login`, managed identity, or environment credentials automatically).
  2. Exchange that for a Video Indexer *access token* by calling the ARM
     `generateAccessToken` endpoint.
  3. Use that access token on every call to api.videoindexer.ai.

This mirrors exactly what the "Explore the portal" web UI does under the
hood -- we're just doing it from code instead of clicking through it.
"""

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from azure.identity import DefaultAzureCredential

import config

ARM_BASE = "https://management.azure.com"
VI_API_BASE = "https://api.videoindexer.ai"


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class VideoIndexerClient:
    def __init__(self):
        self._credential = DefaultAzureCredential()
        self._vi_access_token = None
        self._vi_token_expiry = 0
        self._session = _create_session()

    # ---- auth ----

    def _get_arm_token(self) -> str:
        token = self._credential.get_token("https://management.azure.com/.default")
        return token.token

    def _get_vi_access_token(self, permission: str = "Contributor") -> str:
        """Fetch (and cache) a Video Indexer data-plane access token."""
        if self._vi_access_token and time.time() < self._vi_token_expiry:
            return self._vi_access_token

        arm_token = self._get_arm_token()
        url = (
            f"{ARM_BASE}/subscriptions/{config.VI_SUBSCRIPTION_ID}"
            f"/resourceGroups/{config.VI_RESOURCE_GROUP}"
            f"/providers/Microsoft.VideoIndexer/accounts/{config.VI_ACCOUNT_NAME}"
            f"/generateAccessToken?api-version=2022-08-01"
        )
        resp = self._session.post(
            url,
            headers={"Authorization": f"Bearer {arm_token}"},
            json={"permissionType": permission, "scope": "Account"},
            timeout=30,
        )
        resp.raise_for_status()
        self._vi_access_token = resp.json()["accessToken"]
        self._vi_token_expiry = time.time() + 55 * 60  # tokens last ~1hr, refresh a bit early
        return self._vi_access_token

    # ---- video lifecycle ----

    def upload_video(self, video_name: str, video_url: str = None, file_path: str = None, language: str = "auto") -> str:
        """Uploads a video via URL (preferred) or local file stream. Returns the videoId."""
        access_token = self._get_vi_access_token()
        url = (
            f"{VI_API_BASE}/{config.VI_LOCATION}/Accounts/{config.VI_ACCOUNT_ID}/Videos"
        )
        params = {
            "accessToken": access_token,
            "name": video_name,
            "privacy": "Private",
            "language": language,
            "indexingPreset": "Default",  # matches "Standard video + audio" in the portal
        }
        if video_url:
            params["videoUrl"] = video_url
        
        last_err = None
        for attempt in range(3):
            try:
                if video_url:
                    resp = self._session.post(url, params=params, timeout=60)
                else:
                    with open(file_path, "rb") as f:
                        files = {"file": (video_name, f, "application/octet-stream")}
                        resp = self._session.post(url, params=params, files=files, timeout=300)
                resp.raise_for_status()
                return resp.json()["id"]
            except Exception as e:
                last_err = e
                time.sleep(2)
        raise last_err

    def wait_for_processing(self, video_id: str, poll_seconds: int = 10, timeout_seconds: int = 3600) -> dict:
        """Polls until indexing finishes. Returns the full insights JSON."""
        access_token = self._get_vi_access_token()
        url = (
            f"{VI_API_BASE}/{config.VI_LOCATION}/Accounts/{config.VI_ACCOUNT_ID}"
            f"/Videos/{video_id}/Index"
        )
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                resp = self._session.get(url, params={"accessToken": access_token}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                state = data.get("state")
                if state == "Processed":
                    return data
                if state == "Failed":
                    raise RuntimeError(f"Video Indexer failed to process video {video_id}: {data}")
            except Exception as e:
                if "Failed" in str(e) or "Processed" in str(e):
                    raise
            time.sleep(poll_seconds)
        raise TimeoutError(f"Video {video_id} did not finish processing within {timeout_seconds}s")

    def get_thumbnail_bytes(self, video_id: str, thumbnail_id: str) -> bytes:
        """Fetches the actual JPEG bytes for a keyframe thumbnailId."""
        access_token = self._get_vi_access_token()
        url = (
            f"{VI_API_BASE}/{config.VI_LOCATION}/Accounts/{config.VI_ACCOUNT_ID}"
            f"/Videos/{video_id}/Thumbnails/{thumbnail_id}"
        )
        last_err = None
        for attempt in range(3):
            try:
                resp = self._session.get(url, params={"accessToken": access_token, "format": "Jpeg"}, timeout=30)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise last_err