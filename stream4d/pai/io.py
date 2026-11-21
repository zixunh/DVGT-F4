"""Bounded access to raw PAI archives, without extracting entire chunks."""

from contextlib import contextmanager
import csv
import io
import json
from pathlib import Path
import shutil
import struct
import zipfile
import pyarrow.parquet as pq


class StoredMember(io.RawIOBase):
    def __init__(self, path, info):
        super().__init__()
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError(
                "LiDAR random access requires an unencrypted ZIP_STORED member"
            )
        self.handle = Path(path).open("rb")
        self.handle.seek(info.header_offset)
        header = self.handle.read(30)
        if header[:4] != b"PK\x03\x04":
            self.handle.close()
            raise ValueError("Invalid ZIP local header")
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        self.start = info.header_offset + 30 + name_size + extra_size
        self.size, self.position = info.file_size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid whence")
        pos = offset + (
            0 if whence == 0 else self.position if whence == 1 else self.size
        )
        if pos < 0:
            raise ValueError("Negative seek")
        self.position = pos
        return pos

    def read(self, size=-1):
        left = max(0, self.size - self.position)
        self.handle.seek(self.start + self.position)
        data = self.handle.read(left if size < 0 else min(left, size))
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        if hasattr(self, "handle"):
            self.handle.close()
        super().close()


class RawMini:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.plan = json.loads((self.root / "download_plan.json").read_text())
        if self.plan.get("status") != "complete":
            raise ValueError("Raw mini download is not complete")
        self.clips = [
            json.loads(line)
            for line in (self.root / "clips.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len({x["clip_id"] for x in self.clips}) != len(self.clips):
            raise ValueError("Duplicate clip IDs")
        for clip in self.clips:
            expected = clip["split"] == "train" and not clip["foundation4d_eval"]
            if clip["training_eligible"] != expected:
                raise ValueError("Inconsistent train/benchmark split flags")
        self.features = {
            row["feature"]: row
            for row in csv.DictReader((self.root / "features.csv").open())
        }
        self.calibration = {}
        for feature in self.plan["features"]["calibration"]:
            rows = pq.read_table(self.path(feature)).to_pylist()
            self.calibration[feature] = {}
            for row in rows:
                self.calibration[feature].setdefault(row["clip_id"], []).append(row)

    def path(self, feature):
        return self.root / self.features[feature]["chunk_path"].format(
            chunk_id=self.plan["chunk"]
        )

    def member(self, feature, clip, kind):
        return json.loads(self.features[feature]["clip_files_in_zip"])[kind].format(
            clip_id=clip
        )

    def table(self, feature, clip, kind=None):
        with zipfile.ZipFile(self.path(feature)) as archive:
            data = archive.read(self.member(feature, clip, kind or feature))
        return pq.read_table(io.BytesIO(data))

    def video(self, camera, clip, destination):
        destination = Path(destination) / f"{camera}.mp4"
        with zipfile.ZipFile(self.path(camera)) as archive:
            with (
                archive.open(self.member(camera, clip, "video")) as source,
                destination.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
        return destination

    @contextmanager
    def lidar(self, clip):
        feature = "lidar_top_360fov"
        path = self.path(feature)
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(self.member(feature, clip, "pointclouds"))
        with StoredMember(path, info) as source:
            yield pq.ParquetFile(source)
