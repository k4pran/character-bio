"""Publish an immutable model bundle and a stable manifest to Firebase Storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DEFAULT_MODEL_DIR = "exported/character-ner-onnx-updated"
DEFAULT_PREFIX = "models"
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class ModelFile:
    path: Path
    relative_path: str
    size: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_model_files(model_dir: Path) -> list[ModelFile]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    paths = sorted(
        (path for path in model_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(model_dir).as_posix(),
    )
    if not paths:
        raise ValueError(f"Model directory contains no files: {model_dir}")

    files = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Model bundles cannot contain symlinks: {path}")
        files.append(
            ModelFile(
                path=path,
                relative_path=path.relative_to(model_dir).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return files


def bundle_version(files: Iterable[ModelFile]) -> str:
    """Hash both relative paths and file hashes to identify the whole bundle."""
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file.sha256))
    return digest.hexdigest()


def clean_storage_component(value: str, name: str) -> str:
    value = value.strip().strip("/")
    if not value or value in {".", ".."}:
        raise ValueError(f"{name} must not be empty")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{name} contains an unsafe path component: {value}")
    return value


def storage_path(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def create_manifest(
    *,
    bucket_name: str,
    prefix: str,
    model_name: str,
    version: str,
    files: Iterable[ModelFile],
    published_at: str | None = None,
) -> dict[str, Any]:
    version_root = storage_path(prefix, model_name, "versions", version)
    latest_object = storage_path(prefix, model_name, "latest.json")
    return {
        "schema_version": 1,
        "model_name": model_name,
        "version": version,
        "published_at": published_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bucket": bucket_name,
        "manifest_storage_path": latest_object,
        "version_storage_path": version_root,
        "files": [
            {
                "path": file.relative_path,
                "storage_path": storage_path(version_root, file.relative_path),
                "gs_uri": (
                    f"gs://{bucket_name}/"
                    f"{storage_path(version_root, file.relative_path)}"
                ),
                "size": file.size,
                "sha256": file.sha256,
            }
            for file in files
        ],
    }


def content_type_for(path: Path) -> str:
    if path.suffix.lower() == ".onnx":
        return "application/octet-stream"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def read_remote_manifest(blob: Any) -> dict[str, Any] | None:
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Existing manifest is not valid UTF-8 JSON: {blob.name}"
        ) from error


def upload_bundle(
    *,
    storage_client: Any,
    model_dir: Path,
    bucket_name: str,
    model_name: str | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> tuple[dict[str, Any], int]:
    """Upload missing immutable objects, then atomically point latest.json at them."""
    files = collect_model_files(model_dir)
    version = bundle_version(files)
    model_name = clean_storage_component(model_name or model_dir.name, "model name")
    prefix = clean_storage_component(prefix, "prefix")
    bucket_name = bucket_name.strip()
    if not bucket_name:
        raise ValueError("Firebase Storage bucket must not be empty")

    bucket = storage_client.bucket(bucket_name)
    version_root = storage_path(prefix, model_name, "versions", version)
    uploaded_count = 0

    for file in files:
        object_name = storage_path(version_root, file.relative_path)
        blob = bucket.blob(object_name)
        if blob.exists():
            blob.reload()
            remote_sha256 = (blob.metadata or {}).get("sha256")
            if remote_sha256 != file.sha256:
                raise RuntimeError(
                    "Refusing to overwrite immutable object with missing or "
                    f"different hash metadata: gs://{bucket_name}/{object_name}"
                )
            print(f"unchanged: gs://{bucket_name}/{object_name}")
            continue

        blob.metadata = {
            "sha256": file.sha256,
            "model_name": model_name,
            "model_version": version,
        }
        blob.cache_control = "private, max-age=31536000, immutable"
        blob.chunk_size = UPLOAD_CHUNK_SIZE
        blob.upload_from_filename(
            str(file.path),
            content_type=content_type_for(file.path),
            if_generation_match=0,
            timeout=600,
        )
        uploaded_count += 1
        print(f"uploaded:  gs://{bucket_name}/{object_name}")

    latest_blob = bucket.blob(storage_path(prefix, model_name, "latest.json"))
    existing_manifest = read_remote_manifest(latest_blob)
    if existing_manifest and existing_manifest.get("version") == version:
        print("latest.json already points to this bundle")
        return existing_manifest, uploaded_count

    manifest = create_manifest(
        bucket_name=bucket_name,
        prefix=prefix,
        model_name=model_name,
        version=version,
        files=files,
    )
    latest_blob.cache_control = "no-store"
    latest_blob.upload_from_string(
        json.dumps(manifest, indent=2) + "\n",
        content_type="application/json; charset=utf-8",
        timeout=60,
    )
    print(f"updated:   gs://{bucket_name}/{latest_blob.name}")
    return manifest, uploaded_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a model directory to immutable Firebase Storage paths and "
            "update its stable latest.json manifest."
        )
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--bucket",
        default=os.environ.get("FIREBASE_STORAGE_BUCKET"),
        help="Firebase Storage bucket (or FIREBASE_STORAGE_BUCKET).",
    )
    parser.add_argument(
        "--model-name",
        help="Storage name for the model; defaults to the directory name.",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("FIREBASE_MODEL_PREFIX", DEFAULT_PREFIX),
    )
    parser.add_argument(
        "--manifest-out",
        help="Optionally write the resulting manifest to this local path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hash the bundle and print its manifest without authenticating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bucket:
        raise SystemExit(
            "Firebase bucket is required. Pass --bucket or set "
            "FIREBASE_STORAGE_BUCKET."
        )

    model_dir = Path(args.model_dir)
    model_name = clean_storage_component(
        args.model_name or model_dir.name, "model name"
    )
    prefix = clean_storage_component(args.prefix, "prefix")

    if args.dry_run:
        files = collect_model_files(model_dir)
        manifest = create_manifest(
            bucket_name=args.bucket,
            prefix=prefix,
            model_name=model_name,
            version=bundle_version(files),
            files=files,
        )
    else:
        try:
            from google.cloud import storage
        except ImportError as error:
            raise SystemExit(
                "Publishing dependencies are missing. Run "
                "'python -m pip install -r requirements-publish.txt'."
            ) from error

        manifest, uploaded_count = upload_bundle(
            storage_client=storage.Client(),
            model_dir=model_dir,
            bucket_name=args.bucket,
            model_name=model_name,
            prefix=prefix,
        )
        print(f"new files uploaded: {uploaded_count}")

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
