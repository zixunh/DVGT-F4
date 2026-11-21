"""Convert local verified PAI chunks to bounded, per-frame stream4d inputs."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import time

import cv2
import DracoPy
import numpy as np

from ..config import load_config
from .annotations import DETECTION_CLASSES, Tracks
from .geometry import (
    Rectifier,
    Trajectory,
    binary_occupancy,
    pose_from_row,
    sparse_depth,
    transform,
)
from .io import RawMini

CONVERSION_VERSION = 1


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class Video:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open {path}")
        self.index, self.cached = -1, None

    def get(self, index):
        if index < self.index:
            raise ValueError("Video conversion must advance sequentially")
        if index == self.index:
            return self.cached
        while self.index < index:
            if not self.capture.grab():
                raise ValueError("Video ended before its timestamp table")
            self.index += 1
        ok, self.cached = self.capture.retrieve()
        if not ok:
            raise ValueError("Could not decode camera frame")
        return self.cached

    def close(self):
        self.capture.release()


def nearest_index(times, wanted):
    right = np.searchsorted(times, wanted)
    choices = [i for i in (right - 1, right) if 0 <= i < len(times)]
    return int(min(choices, key=lambda i: abs(times[i] - wanted)))


def decode_lidar(row, trajectory, sensor_to_ego, dimensions, reference_time):
    cloud = DracoPy.decode(row["draco_encoded_pointcloud"])
    attributes = {a["name"]: a["data"] for a in cloud.attributes}
    if "timestamp" not in attributes:
        raise ValueError(
            "Expected current PAI named per-point timestamps, not legacy color encoding"
        )
    times = attributes["timestamp"].reshape(-1).astype(np.int64)
    xyz = np.asarray(cloud.points, dtype=np.float64)
    rig = transform(xyz, sensor_to_ego)
    center = np.array(
        [dimensions["rear_axle_to_bbox_center"], 0, dimensions["height"] / 2]
    )
    half = np.array([dimensions[k] for k in ("length", "width", "height")]) / 2 + 0.5
    good = np.isfinite(rig).all(-1) & (np.abs(rig - center) > half).any(-1)
    good &= (times >= row["spin_start_timestamp"]) & (
        times <= row["spin_end_timestamp"]
    )
    good &= (
        (times >= trajectory.times[0])
        & (times <= trajectory.times[-1])
        & (times <= reference_time)
    )
    rig, times = rig[good], times[good]
    # Equal timestamps share a pose. Only one completed sweep is used; no future sweeps.
    unique, inverse = np.unique(times, return_inverse=True)
    poses = trajectory.at(unique)
    anchor_points = transform(rig, poses[inverse])
    anchor_origins = transform(
        np.broadcast_to(sensor_to_ego[:3, 3], rig.shape), poses[inverse]
    )
    anchor_to_ref = np.linalg.inv(trajectory.at(reference_time))
    return transform(anchor_points, anchor_to_ref), transform(
        anchor_origins, anchor_to_ref
    )


def convert_clip(job):
    raw_root, output, clip, options, fingerprint = job
    cv2.setNumThreads(1)
    started = time.monotonic()
    raw = RawMini(raw_root)
    cid = clip["clip_id"]
    destination = Path(output) / "clips" / cid
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "conversion.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError("Existing clip was converted with different options")
        for record in (destination / "frames.jsonl").read_text().splitlines():
            if not (destination / json.loads(record)["npz"]).is_file():
                raise ValueError(
                    "Completed conversion has a missing frame; use a fresh output directory"
                )
        return report
    cameras = options["cameras"]
    intrinsics = {
        r["camera_name"]: r for r in raw.calibration["camera_intrinsics.offline"][cid]
    }
    extrinsics = {
        r["sensor_name"]: pose_from_row(r)
        for r in raw.calibration["sensor_extrinsics.offline"][cid]
    }
    rectifiers = [
        Rectifier(
            intrinsics[c],
            options["size"],
            25 if "30fov" in c else 60 if "70fov" in c else 100,
        )
        for c in cameras
    ]
    k = np.stack([r.k for r in rectifiers])
    e = np.stack([extrinsics[c] for c in cameras]).astype(np.float32)
    masks = np.stack([r.valid for r in rectifiers])
    timestamps = []
    for c, rect in zip(cameras, rectifiers):
        table = raw.table(c, cid, "frame_timestamps")
        if not np.array_equal(
            table["frame_index"].to_numpy(), np.arange(table.num_rows)
        ):
            raise ValueError("Non-contiguous camera frame indices")
        values = (
            table["timestamp"].to_numpy().astype(np.int64) - rect.shutter_delay_us // 2
        )
        if np.any(np.diff(values) <= 0):
            raise ValueError("Camera timestamps are not strictly increasing")
        timestamps.append(values)
    trajectory = Trajectory(raw.table("egomotion.offline", cid).to_pylist())
    tracks = Tracks(raw.table("obstacle.offline", cid).to_pylist(), trajectory)
    dimensions = raw.calibration["vehicle_dimensions"][cid][0]
    reference_camera = cameras.index("camera_front_wide_120fov")
    counter, skipped = Counter(), Counter()
    records, camera_skews, lidar_ages = [], [], []
    origin = None
    valid_start = max(trajectory.times[0], max(t[0] for t in timestamps), 250000)
    valid_end = min(trajectory.times[-1], min(t[-1] for t in timestamps), 20000000)
    next_time = valid_start
    with (
        tempfile.TemporaryDirectory(prefix="pai-clip-") as scratch,
        raw.lidar(cid) as lidar,
    ):
        timing = lidar.read(columns=["spin_start_timestamp", "spin_end_timestamp"])
        ends = timing["spin_end_timestamp"].to_numpy()
        if (
            np.any(np.diff(ends) <= 0)
            or lidar.metadata.num_row_groups != timing.num_rows
        ):
            raise ValueError("Expected one ordered LiDAR spin per parquet row group")
        videos = [Video(raw.video(c, cid, scratch)) for c in cameras]
        try:
            for wanted in timestamps[reference_camera]:
                if wanted < next_time or wanted > valid_end:
                    continue
                indices = [nearest_index(t, wanted) for t in timestamps]
                actual_times = np.array(
                    [t[i] for t, i in zip(timestamps, indices)], dtype=np.int64
                )
                ref = int(actual_times.max())
                target_indices = [i + options["target_offset_frames"] for i in indices]
                if any(i >= len(t) for t, i in zip(timestamps, target_indices)):
                    break
                target_times = np.array(
                    [t[i] for t, i in zip(timestamps, target_indices)], dtype=np.int64
                )
                if actual_times.min() < valid_start or target_times.max() > valid_end:
                    continue
                skew = int(ref - actual_times.min())
                if skew > options["max_camera_skew_us"]:
                    skipped["camera_skew"] += 1
                    continue
                spin_index = int(
                    np.searchsorted(ends, actual_times.min(), side="right") - 1
                )
                if spin_index < 0 or ref - ends[spin_index] > 200000:
                    skipped["no_recent_completed_lidar_sweep"] += 1
                    continue
                row = lidar.read_row_group(spin_index).to_pylist()[0]
                points, origins = decode_lidar(
                    row, trajectory, extrinsics["lidar_top_360fov"], dimensions, ref
                )
                pose = trajectory.at(ref)
                if origin is None:
                    origin = pose
                world_from_anchor = np.linalg.inv(origin)
                boxes = tracks.at(ref, np.linalg.inv(pose))
                bounds = options["detection_bounds"]
                inside = (
                    (boxes["boxes"][:, :3] >= bounds[:3])
                    & (boxes["boxes"][:, :3] < bounds[3:])
                ).all(-1)
                boxes = {key: value[inside] for key, value in boxes.items()}
                if len(boxes["boxes"]) > options["detection_queries"]:
                    raise ValueError(
                        "More annotated boxes than detection queries; increase detection_queries"
                    )
                images = np.stack(
                    [r.image(v.get(i)) for r, v, i in zip(rectifiers, videos, indices)]
                )
                depths = np.stack(
                    [
                        sparse_depth(points, ki, ei, mask)
                        for ki, ei, mask in zip(k, e, masks)
                    ]
                )
                occupancy = binary_occupancy(
                    points,
                    origins,
                    options["occupancy_bounds"],
                    options["occupancy_shape"],
                )
                arrays = dict(
                    images=images,
                    intrinsics=k,
                    camera_to_ego=e,
                    depth_z=depths,
                    occupancy_binary=occupancy,
                    world_from_ego=(world_from_anchor @ pose).astype(np.float32),
                    image_valid=masks,
                    **boxes,
                )
                if options["target_offset_frames"]:
                    target_rgb = np.stack(
                        [
                            r.image(v.get(i))
                            for r, v, i in zip(rectifiers, videos, target_indices)
                        ]
                    )
                    target_boxes = tracks.at(int(target_times.max()), world_from_anchor)
                    arrays.update(
                        target_rgb=target_rgb,
                        target_intrinsics=k,
                        target_camera_to_world=(
                            world_from_anchor @ trajectory.at(target_times) @ e
                        ).astype(np.float32),
                        target_valid=masks,
                        target_boxes_world=target_boxes["boxes"],
                        target_box_ids=target_boxes["box_ids"],
                    )
                name = f"frame_{len(records):06d}.npz"
                temporary = destination / (name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                temporary.replace(destination / name)
                record = dict(
                    npz=name,
                    scene_id=cid,
                    timestamp_us=ref,
                    camera_timestamps_us=actual_times.tolist(),
                    camera_ids=cameras,
                    dataset="pai",
                    split=clip["split"],
                    training_eligible=clip["training_eligible"],
                    foundation4d_eval=clip["foundation4d_eval"],
                    synthetic=False,
                    dataset_revision=raw.plan["revision"],
                    detection_classes=DETECTION_CLASSES,
                    occupancy_bounds=options["occupancy_bounds"],
                    supervision=[
                        "sparse_lidar_depth",
                        "offline_autolabel_boxes",
                        "lidar_binary_occupancy",
                    ],
                    source_frame_indices=[int(i) for i in indices],
                    source_lidar_spin=int(row["spin_index"]),
                )
                if options["target_offset_frames"]:
                    record.update(
                        target_is_held_out=True,
                        target_camera_indices=list(range(len(cameras))),
                        target_timestamps_us=target_times.tolist(),
                        target_frame_indices=target_indices,
                    )
                records.append(record)
                counter.update(
                    frames=1,
                    boxes=len(boxes["boxes"]),
                    depth_valid_pixels=int((depths > 0).sum()),
                    occupancy_free=int((occupancy == 0).sum()),
                    occupancy_occupied=int((occupancy == 1).sum()),
                    occupancy_unknown=int((occupancy < 0).sum()),
                )
                camera_skews.append(skew)
                lidar_ages.append(int(ref - ends[spin_index]))
                next_time = int(wanted + round(1000000 / options["fps"]))
                if options["max_frames"] and len(records) >= options["max_frames"]:
                    break
        finally:
            for video in videos:
                video.close()
    if not records:
        raise ValueError(f"Clip {cid} produced no valid synchronized frames")
    manifest = destination / "frames.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    report = dict(
        clip_id=cid,
        split=clip["split"],
        training_eligible=clip["training_eligible"],
        fingerprint=fingerprint,
        counts=dict(counter),
        skipped=dict(skipped),
        max_camera_skew_us=max(camera_skews),
        max_lidar_age_us=max(lidar_ages),
        rectified_valid_fraction=masks.mean(axis=(1, 2)).tolist(),
        track_id_map=tracks.id_map,
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="/mnt/raid/dataset/PAI/mini")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/stream4d/pai.yaml")
    parser.add_argument("--size", nargs=2, type=int, default=[96, 160])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Per clip; 0 converts the complete clip",
    )
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument(
        "--target-offset-frames",
        type=int,
        default=1,
        help="Future held-out RGB offset; 0 disables reconstruction targets",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if (
        any(x <= 0 or x % 16 for x in args.size)
        or not 0 < args.fps <= 10
        or args.workers < 1
        or min(args.max_clips, args.max_frames, args.target_offset_frames) < 0
    ):
        parser.error("Invalid size/rate/workers/limits")
    if args.target_offset_frames >= 30 / args.fps:
        parser.error(
            "Held-out offset must be shorter than the source sampling interval"
        )
    if config.heads.detection_classes != len(DETECTION_CLASSES):
        parser.error("PAI requires the explicit 12-class detection configuration")
    raw = RawMini(args.raw_root)
    options = dict(
        version=CONVERSION_VERSION,
        size=args.size,
        fps=args.fps,
        max_frames=args.max_frames,
        target_offset_frames=args.target_offset_frames,
        cameras=raw.plan["features"]["camera"],
        max_camera_skew_us=round(config.stream.max_camera_skew_seconds * 1e6),
        detection_bounds=config.heads.detection_bounds,
        detection_queries=config.heads.detection_queries,
        occupancy_bounds=config.heads.occupancy_bounds,
        occupancy_shape=config.heads.occupancy_shape,
        dataset_revision=raw.plan["revision"],
    )
    fingerprint = hashlib.sha256(
        json.dumps(options, sort_keys=True).encode()
    ).hexdigest()
    root = Path(args.output).resolve()
    if root == raw.root or raw.root in root.parents:
        parser.error("Converted data must use a separate directory from raw data")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "conversion_plan.json"
    if (
        plan_path.exists()
        and json.loads(plan_path.read_text())["fingerprint"] != fingerprint
    ):
        parser.error(
            "Existing output uses different conversion options; choose a new output directory"
        )
    clips = [c for c in raw.clips if args.split == "all" or c["split"] == args.split]
    if args.max_clips:
        selected = clips[: args.max_clips]
        # A smoke subset of >=2 clips should exercise the held-out split too.
        if (
            args.split == "all"
            and args.max_clips >= 2
            and len({c["split"] for c in selected}) == 1
        ):
            other = next((c for c in clips if c["split"] != selected[0]["split"]), None)
            if other is not None:
                selected[-1] = other
        clips = selected
    if not clips:
        parser.error("No clips match the requested split")
    plan = dict(
        status="converting",
        raw_root=str(raw.root),
        fingerprint=fingerprint,
        options=options,
        selected_clip_ids=[c["clip_id"] for c in clips],
        limitations=[
            "Mid-readout global-shutter approximation; no per-row rolling-shutter correction",
            "Fixed rig calibration with bounded camera skew; no image warping across asynchronous views",
            "Single past LiDAR sweep with per-point ego-motion compensation; no moving-object deskew",
            "Sparse measured depth and binary ray occupancy, not Foundation4D dense/semantic labels",
            "Offline cuboids are NVIDIA autolabels, not manual ground truth",
        ],
    )
    write_json(plan_path, plan)
    reports = []
    jobs = [(str(raw.root), str(root), clip, options, fingerprint) for clip in clips]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(convert_clip, jobs):
            reports.append(result)
            print(
                json.dumps(
                    {
                        "event": "converted",
                        "clip_id": result["clip_id"],
                        "split": result["split"],
                        "frames": result["counts"]["frames"],
                        "completed": len(reports),
                        "total": len(jobs),
                    }
                ),
                flush=True,
            )
    for split in sorted({c["split"] for c in clips}):
        path = root / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as output:
            for clip in clips:
                if clip["split"] != split:
                    continue
                cid = clip["clip_id"]
                for line in (
                    (root / "clips" / cid / "frames.jsonl").read_text().splitlines()
                ):
                    record = json.loads(line)
                    record["npz"] = f"clips/{cid}/{record['npz']}"
                    output.write(json.dumps(record) + "\n")
        temporary.replace(path)
    plan.update(
        status="complete",
        converted_clips=len(reports),
        split_clip_counts=dict(Counter(r["split"] for r in reports)),
        split_frame_counts={
            s: sum(r["counts"]["frames"] for r in reports if r["split"] == s)
            for s in {r["split"] for r in reports}
        },
        total_frames=sum(r["counts"]["frames"] for r in reports),
    )
    write_json(plan_path, plan)
    print(
        json.dumps(
            {
                k: plan[k]
                for k in [
                    "status",
                    "converted_clips",
                    "split_clip_counts",
                    "split_frame_counts",
                    "total_frames",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
