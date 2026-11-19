import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[2] / "tools/data_process/download_pai_mini.py"
SPEC = importlib.util.spec_from_file_location("download_pai_mini", SCRIPT)
mini = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mini)


class MiniDownloadTests(unittest.TestCase):
    def write_metadata(self, root, missing_feature=False, invalid=False):
        (root / "metadata").mkdir()
        (root / "splits").mkdir()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"clip_id": "a", "chunk": 0, "clip_is_valid": not invalid},
                    {"clip_id": "b", "chunk": 0, "clip_is_valid": True},
                    {"clip_id": "c", "chunk": 1, "clip_is_valid": True},
                ]
            ),
            root / "clip_index.parquet",
        )
        features = {
            feature: True for group in mini.FEATURES.values() for feature in group
        }
        rows = [{"clip_id": clip, **features} for clip in ["a", "b", "c"]]
        if missing_feature:
            rows[0]["lidar_top_360fov"] = False
        pq.write_table(
            pa.Table.from_pylist(rows), root / "metadata/feature_presence.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"clip_id": "a", "split": "train"},
                    {"clip_id": "b", "split": "test"},
                    {"clip_id": "c", "split": "alpasim"},
                ]
            ),
            root / "splits/alpamayo_aligned.parquet",
        )

    def test_only_requested_chunk_and_original_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_metadata(root)
            selected = mini.select_clips(root, 0, 2)
            self.assertEqual([x["clip_id"] for x in selected], ["a", "b"])
            self.assertEqual([x["split"] for x in selected], ["train", "test"])
            with self.assertRaisesRegex(ValueError, "expected 100"):
                mini.select_clips(root, 0, 100)

    def test_incomplete_or_invalid_clip_is_rejected_before_sensor_download(self):
        for kwargs, message in [
            ({"missing_feature": True}, "lacks"),
            ({"invalid": True}, "invalid"),
        ]:
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_metadata(root, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    mini.select_clips(root, 0, 2)

    def test_foundation4d_eval_excluded_even_if_pai_split_is_train(self):
        api = SimpleNamespace(
            dataset_info=lambda repo: SimpleNamespace(sha="pinned"),
            list_repo_tree=lambda *args, **kwargs: [SimpleNamespace(path="clips/a")],
        )
        selected = [
            {"clip_id": "a", "split": "train"},
            {"clip_id": "b", "split": "test"},
            {"clip_id": "c", "split": "train"},
        ]
        mini.annotate_foundation4d_membership(api, selected)
        self.assertEqual(
            [x["training_eligible"] for x in selected], [False, False, True]
        )

    def test_hash_mismatch_is_not_marked_verified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.zip"
            path.write_bytes(b"corrupt")
            entry = {
                "path": path.name,
                "bytes": 7,
                "sha256": hashlib.sha256(b"correct").hexdigest(),
            }
            with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    mini.download_one(root, "pinned", entry, {})


if __name__ == "__main__":
    unittest.main()
