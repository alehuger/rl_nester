"""GCS download/upload helpers for cloud train and eval workers."""

from __future__ import annotations

from pathlib import Path

from google.cloud import storage


def download_data_from_gcs(bucket_name: str, blob_path: str, local_path: str) -> None:
    print(f"Downloading gs://{bucket_name}/{blob_path} to {local_path}")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded {local_path}")


def download_blob_from_gcs_if_exists(bucket_name: str, blob_path: str, local_path: str) -> bool:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return False
    download_data_from_gcs(bucket_name, blob_path, local_path)
    return True


def download_dir_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> None:
    print(f"Downloading gs://{bucket_name}/{prefix}/* to {local_dir}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix.rstrip("/") + "/")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in blobs:
        local_path = Path(local_dir) / blob.name[len(prefix.rstrip("/") + "/") :]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1
    print(f"Downloaded {count} files to {local_dir}")


def upload_file_to_gcs(bucket_name: str, blob_path: str, local_path: str) -> None:
    print(f"Uploading {local_path} to gs://{bucket_name}/{blob_path}")
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)
