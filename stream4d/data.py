"""Lazy JSONL -> one-frame NPZ/images. No full-video tensor or prediction list.

Float64 timestamps are converted to relative seconds before any tensor casting.
All intrinsics must describe the supplied rectified image resolution.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

FLOAT_FIELDS = {
    "images",
    "intrinsics",
    "camera_to_ego",
    "depth_z",
    "points_ego",
    "world_from_ego",
    "boxes",
    "normals_ego",
    "target_rgb",
    "target_intrinsics",
    "target_camera_to_world",
    "target_boxes_world",
    "target_range",
    "target_non_sky",
}
INT_FIELDS = {"semantics", "box_labels", "box_ids", "occupancy", "target_box_ids"}
BOOL_FIELDS = {"target_valid"}


def load_frame(record, root):
    if "npz" in record:
        with np.load(root / record["npz"], allow_pickle=False) as stored:
            frame = {k: stored[k] for k in stored.files}
    else:
        frame = {}
    for k, v in record.items():
        if k != "npz":
            frame[k] = v
    if "image_paths" in frame:
        frame["images"] = np.stack(
            [
                np.array(Image.open(root / p).convert("RGB")).transpose(2, 0, 1)
                for p in frame.pop("image_paths")
            ]
        )
    for k in FLOAT_FIELDS | INT_FIELDS | BOOL_FIELDS:
        if k not in frame:
            continue
        x = np.asarray(frame[k])
        if k in ("images", "target_rgb") and x.dtype == np.uint8:
            x = x.astype(np.float32) / 255
        dtype = (
            torch.float32
            if k in FLOAT_FIELDS
            else torch.long
            if k in INT_FIELDS
            else torch.bool
        )
        frame[k] = torch.as_tensor(np.array(x, copy=True), dtype=dtype)
    for k in ("scene_id", "target_is_held_out", "synthetic"):
        if k in frame and isinstance(frame[k], np.ndarray) and frame[k].ndim == 0:
            frame[k] = frame[k].item()
    return frame


def iter_frames(manifest):
    path = Path(manifest).resolve()
    root = path.parent
    origin = None
    previous_scene = None
    with path.open(encoding="utf8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            frame = load_frame(record, root)
            scene = str(frame.get("scene_id", "default"))
            frame["scene_id"] = scene
            if scene != previous_scene:
                origin = None
                previous_scene = scene
            if "timestamp_us" in frame:
                stamp = int(frame.pop("timestamp_us"))
                if origin is None:
                    origin = stamp
                frame["timestamp"] = (stamp - origin) * 1e-6
                if "camera_timestamps_us" in frame:
                    # Subtract as int64 first, then convert; preserve microsecond precision.
                    frame["camera_timestamps"] = (
                        np.asarray(frame.pop("camera_timestamps_us"), dtype=np.int64)
                        - origin
                    ).astype(np.float64) * 1e-6
            elif "timestamp" in frame:
                stamp = float(frame["timestamp"])
                if origin is None:
                    origin = stamp
                frame["timestamp"] = stamp - origin
                if "camera_timestamps" in frame:
                    frame["camera_timestamps"] = (
                        np.asarray(frame["camera_timestamps"], dtype=np.float64)
                        - origin
                    )
            else:
                raise ValueError(
                    f"{path}:{line_number}: timestamp or timestamp_us is required"
                )
            yield frame


def to_device(frame, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in frame.items()
    }


def make_synthetic(directory, config, frames=16, views=2, hw=(32, 48)):
    """Analytic plane + moving cuboid; deterministic fixtures, never benchmark data.

    Render targets are a separate camera viewpoint, not copies of source images.
    All world/ego axes FLU; source frame moves forward. Labels are synthetic.
    """
    from .geometry import (
        rdf_to_flu,
        camera_rays,
        unproject_depth,
        box_poses,
        invert_pose,
    )

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    h, w = hw
    k = torch.tensor(
        [[w * 0.9, 0.0, (w - 1) / 2], [0.0, w * 0.9, (h - 1) / 2], [0.0, 0.0, 1.0]]
    ).repeat(views, 1, 1)
    e = rdf_to_flu(k).repeat(views, 1, 1)
    e[:, 1, 3] = torch.linspace(-0.5, 0.5, views)
    e[:, 2, 3] = 1.5
    boxes_world = torch.tensor([[9.0, -0.8, 0.8, 2.0, 1.5, 1.6, 0.0, 0.0, 0.2]])

    def observe(camera_to_world, box):
        rays = camera_rays(k, (h, w))
        direction = torch.einsum("vij,vhwj->vhwi", camera_to_world[:, :3, :3], rays)
        origin = camera_to_world[:, None, None, :3, 3]
        t = (18 - origin[..., 0]) / direction[..., 0].clamp_min(0.01)
        plane = origin + direction * t[..., None]
        c = 0.5 + 0.25 * torch.sin(plane * plane.new_tensor([0.1, 1.1, 1.7]))
        sem = torch.ones((views, h, w), dtype=torch.long)
        actor_inv = invert_pose(box_poses(box))[0]
        local_origin = (
            torch.einsum(
                "ij,vhwj->vhwi", actor_inv[:3, :3], origin.expand(-1, h, w, -1)
            )
            + actor_inv[:3, 3]
        )
        local_dir = torch.einsum("ij,vhwj->vhwi", actor_inv[:3, :3], direction)
        safe = torch.where(
            local_dir.abs() < 1e-6, torch.full_like(local_dir, 1e-6), local_dir
        )
        limits = box[0, 3:6] / 2
        a = (-limits - local_origin) / safe
        b = (limits - local_origin) / safe
        entry = torch.minimum(a, b).amax(-1)
        leave = torch.maximum(a, b).amin(-1)
        hit = (entry > 0) & (leave >= entry) & (entry < t)
        t = torch.where(hit, entry, t)
        c = torch.where(hit[..., None], c.new_tensor([0.8, 0.2, 0.1]), c)
        sem = torch.where(hit, 2, sem)
        points = origin + direction * t[..., None]
        return c.permute(0, 3, 1, 2).clamp(0, 1), t, sem, points

    manifest = root / "frames.jsonl"
    with manifest.open("w") as out:
        for index in range(frames):
            timestamp = index * 0.5
            pose = torch.eye(4)
            pose[0, 3] = index * 0.05
            actor = boxes_world.clone()
            actor[:, 1] += timestamp * 0.2
            rgb, depth, sem, points = observe(pose[None] @ e, actor)
            target_e = e.clone()
            target_e[:, 1, 3] += 0.15
            target_rgb, target_depth, _, _ = observe(pose[None] @ target_e, actor)
            box = actor.clone()
            box[:, :3] -= pose[:3, 3]
            occ = torch.full(tuple(config.heads.occupancy_shape), -1, dtype=torch.long)
            nx, ny, nz = occ.shape
            # Known free interior and one labeled occupied region; boundary remains unknown.
            occ[1:-1, 1:-1, :] = 0
            occ[nx // 2, ny // 2, : max(1, nz // 2)] = 1
            data = dict(
                images=rgb,
                intrinsics=k,
                camera_to_ego=e,
                depth_z=depth,
                points_ego=unproject_depth(depth, k, e),
                semantics=sem,
                boxes=box,
                box_labels=torch.zeros(1, dtype=torch.long),
                box_ids=torch.ones(1, dtype=torch.long),
                occupancy=occ,
                world_from_ego=pose,
                target_rgb=target_rgb,
                target_intrinsics=k,
                target_camera_to_world=pose[None] @ target_e,
                target_range=target_depth * camera_rays(k, (h, w)).norm(dim=-1),
                target_non_sky=torch.ones_like(target_depth),
                target_boxes_world=actor,
                target_box_ids=torch.ones(1, dtype=torch.long),
            )
            name = f"frame_{index:06d}.npz"
            np.savez_compressed(
                root / name, **{key: value.numpy() for key, value in data.items()}
            )
            out.write(
                json.dumps(
                    dict(
                        npz=name,
                        timestamp_us=1700000000000000 + int(timestamp * 1e6),
                        scene_id="synthetic",
                        synthetic=True,
                        target_is_held_out=True,
                        camera_ids=[f"camera_{i}" for i in range(views)],
                    )
                )
                + "\n"
            )
    return manifest
