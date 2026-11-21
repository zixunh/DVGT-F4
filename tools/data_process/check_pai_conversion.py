"""Audit every converted frame, including train/test and held-out RGB isolation."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from stream4d.pai.annotations import DETECTION_CLASSES


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(root):
    root = Path(root).resolve()
    plan = json.loads((root / "conversion_plan.json").read_text())
    require(plan["status"] == "complete", "Conversion has not completed")
    options = plan["options"]
    raw = Path(plan["raw_root"])
    clips = {
        r["clip_id"]: r
        for r in map(json.loads, (raw / "clips.jsonl").read_text().splitlines())
    }
    counts, split_frames = Counter(), Counter()
    scene_splits, previous, signatures = {}, {}, {}
    source_indices, target_indices = defaultdict(set), defaultdict(set)
    files, manifest_hashes = set(), {}
    max_skew = 0
    for split in sorted(plan["split_frame_counts"]):
        path = root / f"{split}.jsonl"
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        closed_scenes, active_scene = set(), None
        for record in map(json.loads, path.read_text().splitlines()):
            cid = record["scene_id"]
            label = f"{split}/{cid}/{record['npz']}"
            require(
                cid in clips and cid in plan["selected_clip_ids"],
                f"Unexpected clip: {label}",
            )
            require(
                record["split"] == clips[cid]["split"] == split,
                f"Official split mismatch: {label}",
            )
            require(
                record["training_eligible"] == clips[cid]["training_eligible"]
                and record["foundation4d_eval"] == clips[cid]["foundation4d_eval"],
                f"Membership mismatch: {label}",
            )
            if split == "train":
                require(
                    record["training_eligible"] and not record["foundation4d_eval"],
                    f"Prohibited training clip: {label}",
                )
            require(
                record["dataset"] == "pai"
                and record["dataset_revision"] == options["dataset_revision"]
                and not record["synthetic"],
                f"Dataset provenance mismatch: {label}",
            )
            require(
                record["detection_classes"] == DETECTION_CLASSES
                and record["occupancy_bounds"] == options["occupancy_bounds"],
                f"Label protocol mismatch: {label}",
            )
            require(
                record["camera_ids"] == options["cameras"],
                f"Camera order changed: {label}",
            )
            if active_scene != cid:
                require(cid not in closed_scenes, f"Interleaved scene: {label}")
                if active_scene is not None:
                    closed_scenes.add(active_scene)
                active_scene = cid
            require(
                cid not in scene_splits or scene_splits[cid] == split,
                f"Scene appears in multiple splits: {label}",
            )
            scene_splits[cid] = split
            stamp = record["timestamp_us"]
            require(
                cid not in previous or stamp > previous[cid],
                f"Nonincreasing time: {label}",
            )
            times = np.asarray(record["camera_timestamps_us"])
            require(
                len(times) == len(options["cameras"]) and np.all(times <= stamp),
                f"Future source camera: {label}",
            )
            skew = int(stamp - times.min())
            require(
                skew <= options["max_camera_skew_us"], f"Camera skew too large: {label}"
            )
            max_skew = max(max_skew, skew)
            file = (root / record["npz"]).resolve()
            require(
                file.is_relative_to(root / "clips" / cid) and file not in files,
                f"Invalid/duplicate frame path: {label}",
            )
            files.add(file)
            with np.load(file, allow_pickle=False) as frame:
                views, (h, w) = len(options["cameras"]), options["size"]
                require(
                    frame["images"].shape == (views, 3, h, w)
                    and frame["images"].dtype == np.uint8,
                    f"Invalid RGB: {label}",
                )
                require(
                    frame["depth_z"].shape == (views, h, w)
                    and np.isfinite(frame["depth_z"]).all()
                    and (frame["depth_z"] >= 0).all(),
                    f"Invalid depth: {label}",
                )
                require(
                    frame["image_valid"].shape == (views, h, w)
                    and (frame["depth_z"][~frame["image_valid"]] == 0).all(),
                    f"Unmasked depth: {label}",
                )
                require(
                    frame["intrinsics"].shape == (views, 3, 3)
                    and frame["camera_to_ego"].shape == (views, 4, 4),
                    f"Invalid calibration shape: {label}",
                )
                signature = hashlib.sha256(
                    frame["intrinsics"].tobytes() + frame["camera_to_ego"].tobytes()
                ).hexdigest()
                require(
                    cid not in signatures or signatures[cid] == signature,
                    f"Calibration changed within scene: {label}",
                )
                signatures[cid] = signature
                if cid not in previous:
                    require(
                        np.allclose(frame["world_from_ego"], np.eye(4), atol=1e-5),
                        f"World origin is not first ego frame: {label}",
                    )
                boxes, labels = frame["boxes"], frame["box_labels"]
                require(
                    boxes.shape == (len(labels), 9)
                    and np.isfinite(boxes).all()
                    and (boxes[:, 3:6] > 0).all(),
                    f"Invalid boxes: {label}",
                )
                require(
                    np.all((labels >= 0) & (labels < len(DETECTION_CLASSES))),
                    f"Unmapped class: {label}",
                )
                require(
                    len(np.unique(frame["box_ids"])) == len(boxes)
                    and frame["box_velocity_valid"].shape == (len(boxes),),
                    f"Invalid track/velocity masks: {label}",
                )
                occ = frame["occupancy_binary"]
                require(
                    occ.shape == tuple(options["occupancy_shape"])
                    and np.isin(occ, [-1, 0, 1]).all(),
                    f"Invalid occupancy: {label}",
                )
                for c, index in enumerate(record["source_frame_indices"]):
                    source_indices[cid, c].add(index)
                if options["target_offset_frames"]:
                    require(
                        record["target_is_held_out"]
                        and frame["target_rgb"].shape == frame["images"].shape
                        and frame["target_valid"].shape == (views, h, w),
                        f"Invalid held-out target: {label}",
                    )
                    require(
                        np.all(np.asarray(record["target_timestamps_us"]) > times),
                        f"Target must be a different future frame: {label}",
                    )
                    require(
                        frame["target_camera_to_world"].shape == (views, 4, 4)
                        and np.isfinite(frame["target_camera_to_world"]).all(),
                        f"Invalid target pose: {label}",
                    )
                    for c, (src, target) in enumerate(
                        zip(
                            record["source_frame_indices"],
                            record["target_frame_indices"],
                        )
                    ):
                        require(
                            target - src == options["target_offset_frames"],
                            f"Incorrect target index: {label}",
                        )
                        target_indices[cid, c].add(target)
                counts.update(
                    frames=1,
                    boxes=len(boxes),
                    image_pixels=frame["depth_z"].size,
                    measured_depth_pixels=int((frame["depth_z"] > 0).sum()),
                    occupancy_free=int((occ == 0).sum()),
                    occupancy_occupied=int((occ == 1).sum()),
                    occupancy_unknown=int((occ < 0).sum()),
                    npz_bytes=file.stat().st_size,
                )
            split_frames[split] += 1
            previous[cid] = stamp
            if counts["frames"] % 500 == 0:
                print(
                    json.dumps({"event": "audited", "frames": counts["frames"]}),
                    flush=True,
                )
    require(
        dict(split_frames) == plan["split_frame_counts"],
        "Frame counts differ from conversion plan",
    )
    require(
        set(scene_splits) == set(plan["selected_clip_ids"]),
        "Clip coverage differs from conversion plan",
    )
    require(
        all(
            not values.intersection(source_indices[key])
            for key, values in target_indices.items()
        ),
        "Held-out target RGB also appears as source input",
    )
    report = dict(
        status="passed",
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        fingerprint=plan["fingerprint"],
        clips=len(scene_splits),
        split_clips=dict(Counter(scene_splits.values())),
        split_frames=dict(split_frames),
        counts=dict(counts),
        depth_valid_fraction=counts["measured_depth_pixels"] / counts["image_pixels"],
        max_camera_skew_us=max_skew,
        manifest_sha256=manifest_hashes,
        heldout_rgb_disjoint_from_all_source_frames=True,
        official_split_verified=True,
    )
    output = root / "validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data)), flush=True)
