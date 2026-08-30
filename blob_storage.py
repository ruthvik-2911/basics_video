# blob_storage.py
from datetime import datetime, timedelta, timezone
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

import config

_blob_service = None


def _get_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(config.BLOB_CONNECTION_STRING)
    return _blob_service


def upload_raw_video(local_path: str, blob_name: str) -> str:
    client = _get_service().get_blob_client(config.BLOB_CONTAINER_RAW_VIDEOS, blob_name)
    with open(local_path, "rb") as f:
        client.upload_blob(f, overwrite=True, max_concurrency=4)
    return client.url


def get_blob_sas_url(blob_name: str) -> str:
    """Generates a temporary SAS read URL for Video Indexer to ingest directly from Blob Storage."""
    service = _get_service()
    sas_token = generate_blob_sas(
        account_name=service.account_name,
        container_name=config.BLOB_CONTAINER_RAW_VIDEOS,
        blob_name=blob_name,
        account_key=service.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=4)
    )
    return f"https://{service.account_name}.blob.core.windows.net/{config.BLOB_CONTAINER_RAW_VIDEOS}/{blob_name}?{sas_token}"


def upload_keyframe_image(video_id: str, thumbnail_id: str, image_bytes: bytes) -> str:
    blob_name = f"{video_id}/{thumbnail_id}.jpg"
    client = _get_service().get_blob_client(config.BLOB_CONTAINER_KEYFRAMES, blob_name)
    client.upload_blob(image_bytes, overwrite=True, content_type="image/jpeg")
    return client.url


def download_video_to_temp(blob_name: str, dest_path: str):
    """Used later at query time for the exact on-demand ffmpeg frame grab."""
    client = _get_service().get_blob_client(config.BLOB_CONTAINER_RAW_VIDEOS, blob_name)
    with open(dest_path, "wb") as f:
        f.write(client.download_blob().readall())