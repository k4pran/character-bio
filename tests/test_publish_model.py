import tempfile
import unittest
from pathlib import Path

from src.publish_model import (
    bundle_version,
    collect_model_files,
    create_manifest,
    upload_bundle,
)


class FakeBlob:
    def __init__(self, name):
        self.name = name
        self.metadata = None
        self.cache_control = None
        self.chunk_size = None
        self.content = None
        self.upload_calls = 0

    def exists(self):
        return self.content is not None

    def reload(self):
        return None

    def download_as_bytes(self):
        return self.content

    def upload_from_filename(self, path, **_kwargs):
        self.content = Path(path).read_bytes()
        self.upload_calls += 1

    def upload_from_string(self, content, **_kwargs):
        self.content = content.encode("utf-8")
        self.upload_calls += 1


class FakeBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        return self.blobs.setdefault(name, FakeBlob(name))


class FakeStorageClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, FakeBucket())


class PublishModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name) / "bundle"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "nested").mkdir()
        (self.model_dir / "nested" / "model.onnx").write_bytes(b"model-v1")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bundle_version_changes_with_file_content(self):
        first = bundle_version(collect_model_files(self.model_dir))
        (self.model_dir / "nested" / "model.onnx").write_bytes(b"model-v2")
        second = bundle_version(collect_model_files(self.model_dir))
        self.assertNotEqual(first, second)

    def test_manifest_uses_stable_latest_and_immutable_version_paths(self):
        files = collect_model_files(self.model_dir)
        version = bundle_version(files)
        manifest = create_manifest(
            bucket_name="example.firebasestorage.app",
            prefix="models",
            model_name="character-ner",
            version=version,
            files=files,
            published_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(
            manifest["manifest_storage_path"],
            "models/character-ner/latest.json",
        )
        self.assertTrue(
            all(f"/versions/{version}/" in item["storage_path"]
                for item in manifest["files"])
        )

    def test_second_upload_skips_unchanged_files_and_manifest(self):
        client = FakeStorageClient()
        first_manifest, first_count = upload_bundle(
            storage_client=client,
            model_dir=self.model_dir,
            bucket_name="example.firebasestorage.app",
            model_name="character-ner",
        )
        second_manifest, second_count = upload_bundle(
            storage_client=client,
            model_dir=self.model_dir,
            bucket_name="example.firebasestorage.app",
            model_name="character-ner",
        )

        self.assertEqual(first_count, 2)
        self.assertEqual(second_count, 0)
        self.assertEqual(first_manifest, second_manifest)
        latest = client.bucket(
            "example.firebasestorage.app"
        ).blob("models/character-ner/latest.json")
        self.assertEqual(latest.upload_calls, 1)


if __name__ == "__main__":
    unittest.main()
