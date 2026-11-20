#!/usr/bin/env python3
"""Download one PAI chunk with six cameras and geometry/actor supervision.

The plan is pinned to a dataset commit. Raw archives remain packed, and the
selected clip index retains official split labels. No dataset-wide snapshot is
downloaded. Requires huggingface_hub and pyarrow; uses an existing HF login.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import threading

LOG_LOCK = threading.Lock()

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
FEATURES = {
    "camera": [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_wide_120fov",
        "camera_front_tele_30fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
    ],
    "lidar": ["lidar_top_360fov"],
    "calibration": [
        "camera_intrinsics.offline",
        "sensor_extrinsics.offline",
        "lidar_intrinsics.offline",
        "vehicle_dimensions",
    ],
    "labels": ["egomotion.offline", "obstacle.offline"],
}
METADATA = [
    "features.csv",
    "clip_index.parquet",
    "metadata/data_collection.parquet",
    "metadata/feature_presence.parquet",
    "splits/alpamayo_aligned.parquet",
    "README.md",
]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def log(event, **fields):
    with LOG_LOCK:
        print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def wanted_files(chunk):
    result = list(METADATA)
    for directory, features in FEATURES.items():
        for feature in features:
            extension = "parquet" if directory == "calibration" else "zip"
            result.append(
                f"{directory}/{feature}/{feature}.chunk_{chunk:04d}.{extension}"
            )
    return result


def public_file_metadata(api, revision, paths):
    """The public directory listing exposes sizes even before gated-file access."""
    parents = {}
    for path in paths:
        parents.setdefault(str(Path(path).parent), set()).add(path)

    def read_directory(item):
        parent, wanted = item
        found = []
        for entry in api.list_repo_tree(
            REPO,
            path_in_repo="" if parent == "." else parent,
            revision=revision,
            repo_type="dataset",
        ):
            if entry.path not in wanted:
                continue
            found.append(
                {
                    "path": entry.path,
                    "bytes": entry.size,
                    "sha256": getattr(getattr(entry, "lfs", None), "sha256", None),
                }
            )
            if len(found) == len(wanted):
                break
        missing = wanted - {entry["path"] for entry in found}
        if missing:
            raise RuntimeError(f"Files absent from pinned revision: {sorted(missing)}")
        return found

    files = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entries in pool.map(read_directory, parents.items()):
            files.extend(entries)
    return sorted(files, key=lambda entry: entry["path"])


def refresh_authorized_hashes(api, plan):
    """Replace the masked LFS hashes returned for gated files before login."""
    import re

    entries = api.get_paths_info(
        REPO,
        [entry["path"] for entry in plan["files"]],
        repo_type="dataset",
        revision=plan["revision"],
    )
    resolved = {entry.path: entry for entry in entries}
    for entry in plan["files"]:
        remote = resolved.get(entry["path"])
        if remote is None or remote.size != entry["bytes"]:
            raise RuntimeError(f"Pinned file metadata changed: {entry['path']}")
        lfs = getattr(remote, "lfs", None)
        digest = getattr(lfs, "sha256", None)
        if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
            raise RuntimeError(f"Authorized SHA256 unavailable: {entry['path']}")
        entry["sha256"] = digest


def wait_for_access(seconds):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    deadline = time.monotonic() + seconds
    while True:
        try:
            HfApi().auth_check(repo_id=REPO, repo_type="dataset")
            return
        except HfHubHTTPError as error:
            status = getattr(error.response, "status_code", None)
            if not isinstance(error, GatedRepoError) and status not in (401, 403):
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "PAI access denied. Accept access on the dataset page, then run "
                    "`.venv/bin/hf auth login` as this machine's current user. "
                    "No token should be pasted into chat."
                ) from None
            log("waiting_for_hf_login", http_status=status)
            time.sleep(min(15, max(0, deadline - time.monotonic())))


def clip_key(table):
    if "clip_id" in table.column_names:
        return "clip_id"
    metadata = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
    indices = [key for key in metadata.get("index_columns", []) if isinstance(key, str)]
    if len(indices) == 1 and indices[0] in table.column_names:
        return indices[0]
    raise ValueError(f"Cannot identify clip index in {table.column_names}")


def select_clips(root, chunk, expected):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    index = pq.read_table(root / "clip_index.parquet")
    key = clip_key(index)
    selected = index.filter(pc.equal(index["chunk"], chunk))
    records = selected.to_pylist()
    if len(records) != expected:
        raise ValueError(
            f"Chunk {chunk} has {len(records)} clips, expected {expected}. "
            "No sensor archives have been downloaded."
        )
    if any(record.get("clip_is_valid") is not True for record in records):
        raise ValueError(f"Chunk {chunk} includes invalid clips; choose another chunk.")
    clip_ids = [str(record[key]) for record in records]
    features = pq.read_table(root / "metadata/feature_presence.parquet")
    feature_key = clip_key(features)
    features = features.filter(
        pc.is_in(features[feature_key], value_set=pa.array(clip_ids))
    )
    feature_rows = {str(row[feature_key]): row for row in features.to_pylist()}
    requested = [feature for group in FEATURES.values() for feature in group]
    missing_columns = set(requested) - set(features.column_names)
    if missing_columns:
        raise ValueError(f"Feature availability schema lacks {sorted(missing_columns)}")
    for clip_id in clip_ids:
        absent = [
            name
            for name in requested
            if feature_rows.get(clip_id, {}).get(name) is not True
        ]
        if absent:
            raise ValueError(f"Clip {clip_id} lacks {absent}; choose another chunk.")
    splits = pq.read_table(root / "splits/alpamayo_aligned.parquet")
    split_key = clip_key(splits)
    splits = splits.filter(pc.is_in(splits[split_key], value_set=pa.array(clip_ids)))
    split_rows = {str(row[split_key]): row["split"] for row in splits.to_pylist()}
    return [
        {
            "clip_id": clip_id,
            "chunk": chunk,
            "split": split_rows.get(clip_id, "unknown"),
        }
        for clip_id in sorted(clip_ids)
    ]


def annotate_foundation4d_membership(api, clips):
    """Keep benchmark membership separate from PAI's independent split labels."""
    repo = "nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D"
    revision = api.dataset_info(repo).sha
    evaluation_ids = {
        Path(entry.path).name
        for entry in api.list_repo_tree(
            repo, path_in_repo="clips", repo_type="dataset", revision=revision
        )
    }
    if not evaluation_ids:
        raise RuntimeError("Foundation4D evaluation clip listing is empty.")
    for record in clips:
        record["foundation4d_eval"] = record["clip_id"] in evaluation_ids
        record["training_eligible"] = (
            record["split"] == "train" and not record["foundation4d_eval"]
        )
    return {"repo_id": repo, "revision": revision}


def download_one(root, revision, entry, previous, range_workers=0):
    from huggingface_hub import hf_hub_download

    path = root / entry["path"]
    if path.is_file() and entry["path"] in previous:
        stat = path.stat()
        old = previous[entry["path"]]
        if (
            stat.st_size == entry["bytes"] == old.get("bytes")
            and stat.st_mtime_ns == old.get("mtime_ns")
            and old.get("expected_sha256") == entry["sha256"]
            and old.get("sha256")
        ):
            log("verified_file_reused", path=entry["path"])
            return old
    log("downloading", path=entry["path"], bytes=entry["bytes"])
    ranged = range_workers > 0 and entry["bytes"] >= 256 * 1024**2
    if ranged:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        from pai_http_range import download_ranges

        cache = root / ".cache/huggingface/download" / Path(entry["path"]).parent
        prefixes = [
            p
            for p in cache.glob(f"*.{entry['sha256']}.*.incomplete")
            if p.stat().st_size <= entry["bytes"]
            and p.stat().st_blocks * 512 >= p.stat().st_size
        ]
        prefix = max(prefixes, key=lambda p: p.stat().st_size, default=None)
        filename = download_ranges(
            hf_hub_url(REPO, entry["path"], repo_type="dataset", revision=revision),
            build_hf_headers(),
            root / entry["path"],
            entry["bytes"],
            entry["sha256"],
            workers=range_workers
            if entry["bytes"] >= 8 * 1024**3
            else min(range_workers, 4),
            prefix_file=prefix,
            report=lambda done, total: log(
                "range_progress",
                path=entry["path"],
                completed_bytes=done,
                total_bytes=total,
            ),
        )
    else:
        filename = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            revision=revision,
            filename=entry["path"],
            local_dir=root,
        )
    stat = Path(filename).stat()
    if stat.st_size != entry["bytes"]:
        raise RuntimeError(f"Unexpected size for {entry['path']}")
    if ranged:
        # The range transport verifies the complete SHA256 while assembling.
        observed = entry["sha256"]
    else:
        digest = hashlib.sha256()
        with open(filename, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        observed = digest.hexdigest()
    if entry["sha256"] and entry["sha256"] != observed:
        raise RuntimeError(
            f"SHA256 mismatch for {entry['path']}; remove this file before retrying."
        )
    log("verified", path=entry["path"], bytes=stat.st_size)
    return {
        "path": entry["path"],
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": observed,
        "expected_sha256": entry["sha256"],
    }


def main():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--chunk", type=int, help="Defaults to saved plan chunk, or 0 for a new plan."
    )
    parser.add_argument("--expected-clips", type=int, default=100)
    parser.add_argument(
        "--revision", help="Defaults to saved plan revision, or current main."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Without this flag, only write the size plan.",
    )
    parser.add_argument("--wait-for-auth", type=int, default=0, metavar="SECONDS")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--range-workers",
        type=int,
        default=0,
        help="Parallel HTTP ranges for files >=256 MiB; 0 uses the Hub transport.",
    )
    args = parser.parse_args()
    if (
        (args.chunk is not None and args.chunk < 0)
        or args.expected_clips < 1
        or args.workers < 1
        or args.range_workers < 0
        or args.wait_for_auth < 0
    ):
        parser.error("Invalid count, chunk, worker or wait parameter.")
    disable_progress_bars()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    plan_path = root / "download_plan.json"
    if args.chunk is None:
        args.chunk = (
            json.loads(plan_path.read_text())["chunk"] if plan_path.exists() else 0
        )
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (
            plan["repo_id"] != REPO
            or plan["chunk"] != args.chunk
            or plan["requested_clips"] != args.expected_clips
        ):
            raise ValueError(
                "Output directory contains a different download plan; use another directory."
            )
        if args.revision and args.revision != plan["revision"]:
            raise ValueError("Cannot change the revision of an existing download plan.")
    else:
        revision = api.dataset_info(REPO, revision=args.revision).sha
        entries = public_file_metadata(api, revision, wanted_files(args.chunk))
        plan = {
            "repo_id": REPO,
            "revision": revision,
            "chunk": args.chunk,
            "requested_clips": args.expected_clips,
            "features": FEATURES,
            "files": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "status": "planned",
        }
        write_json(plan_path, plan)
    log(
        "plan",
        revision=plan["revision"],
        files=len(plan["files"]),
        total_GiB=round(plan["total_bytes"] / 1024**3, 3),
    )
    if not args.download:
        return
    wait_for_access(args.wait_for_auth)
    refresh_authorized_hashes(api, plan)
    write_json(plan_path, plan)
    remaining = sum(
        max(
            0,
            entry["bytes"]
            - (
                (root / entry["path"]).stat().st_size
                if (root / entry["path"]).is_file()
                else 0
            ),
        )
        for entry in plan["files"]
    )
    range_reserve = (
        sum(
            entry["bytes"] for entry in plan["files"] if entry["bytes"] >= 256 * 1024**2
        )
        if args.range_workers
        else 0
    )
    if shutil.disk_usage(root).free < remaining + range_reserve + 5 * 1024**3:
        raise RuntimeError(
            "Not enough free space for the remaining files plus a 5 GiB reserve."
        )
    verified_path = root / "verified_files.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    for entry in plan["files"]:
        if entry["path"] in METADATA:
            verified[entry["path"]] = download_one(
                root, plan["revision"], entry, verified
            )
            write_json(verified_path, verified)
    clips = select_clips(root, args.chunk, args.expected_clips)
    plan["foundation4d_split_source"] = annotate_foundation4d_membership(api, clips)
    (root / "clips.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in clips)
    )
    counts = {}
    for record in clips:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    plan.update(
        status="downloading",
        actual_clips=len(clips),
        split_counts=counts,
        training_eligible_clips=sum(record["training_eligible"] for record in clips),
    )
    write_json(plan_path, plan)
    sensor_files = [entry for entry in plan["files"] if entry["path"] not in METADATA]
    sensor_files.sort(key=lambda entry: entry["bytes"], reverse=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                root,
                plan["revision"],
                entry,
                dict(verified),
                args.range_workers,
            ): entry["path"]
            for entry in sensor_files
        }
        for future in as_completed(futures):
            record = future.result()
            verified[record["path"]] = record
            write_json(verified_path, verified)
    plan["status"] = "complete"
    plan["verified_files"] = len(verified)
    write_json(plan_path, plan)
    log(
        "complete",
        clips=len(clips),
        files=len(verified),
        output=str(root),
        split_counts=counts,
    )


if __name__ == "__main__":
    main()
