"""PAI offline autolabels -> metric boxes, stable IDs and absolute velocities."""

from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import transform

DETECTION_CLASSES = [
    "automobile",
    "heavy_truck",
    "bus",
    "train_or_tram_car",
    "trolley_bus",
    "other_vehicle",
    "trailer",
    "person",
    "stroller",
    "rider",
    "animal",
    "protruding_object",
]


class Tracks:
    def __init__(self, rows, trajectory):
        grouped = defaultdict(list)
        self.discarded_outside_pose_support = 0
        for row in rows:
            if row["reference_frame"] != "rig":
                raise ValueError(
                    "Only verified PAI rig-reference cuboids are supported"
                )
            if row["label_class"] not in DETECTION_CLASSES:
                raise ValueError(f"Unmapped PAI class: {row['label_class']}")
            reference = row["reference_frame_timestamp_us"]
            if not trajectory.times[0] <= reference <= trajectory.times[-1]:
                self.discarded_outside_pose_support += 1
                continue
            pose = trajectory.at(reference)
            xyz = transform(
                np.array([row[k] for k in ("center_x", "center_y", "center_z")]), pose
            )
            rotation = (
                pose[:3, :3]
                @ Rotation.from_quat(
                    [
                        row[k]
                        for k in (
                            "orientation_x",
                            "orientation_y",
                            "orientation_z",
                            "orientation_w",
                        )
                    ]
                ).as_matrix()
            )
            grouped[str(row["track_id"])].append(
                (
                    int(row["timestamp_us"]),
                    xyz,
                    rotation,
                    [row[k] for k in ("size_x", "size_y", "size_z")],
                    DETECTION_CLASSES.index(row["label_class"]),
                )
            )
        self.tracks = []
        self.id_map = {key: i + 1 for i, key in enumerate(sorted(grouped))}
        for key in sorted(grouped):
            values = sorted(grouped[key], key=lambda x: x[0])
            times = np.array([v[0] for v in values], dtype=np.int64)
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"Duplicate/nonincreasing cuboid timestamps for {key}")
            self.tracks.append(
                (
                    self.id_map[key],
                    times,
                    np.stack([v[1] for v in values]),
                    np.stack([v[2] for v in values]),
                    np.array([v[3] for v in values]),
                    np.array([v[4] for v in values]),
                )
            )

    def at(
        self,
        timestamp,
        output_from_anchor,
        max_gap_us=250000,
        nearest_tolerance_us=50000,
    ):
        boxes, labels, ids, velocity_valid = [], [], [], []
        for track_id, times, positions, rotations, sizes, classes in self.tracks:
            right = int(np.searchsorted(times, timestamp))
            near = min(max(right, 0), len(times) - 1)
            if right > 0 and abs(times[right - 1] - timestamp) < abs(
                times[near] - timestamp
            ):
                near = right - 1
            bracket = (
                0 < right < len(times) and times[right] - times[right - 1] <= max_gap_us
            )
            if not bracket and abs(times[near] - timestamp) > nearest_tolerance_us:
                continue
            velocity = np.zeros(3)
            if bracket:
                left = right - 1
                gap = times[right] - times[left]
                alpha = (timestamp - times[left]) / gap
                center = (1 - alpha) * positions[left] + alpha * positions[right]
                size = (1 - alpha) * sizes[left] + alpha * sizes[right]
                rotation = Slerp(
                    [0, 1], Rotation.from_matrix(rotations[[left, right]])
                )([alpha]).as_matrix()[0]
                velocity = (positions[right] - positions[left]) / (gap * 1e-6)
            else:
                center, size, rotation = positions[near], sizes[near], rotations[near]
            center = transform(center, output_from_anchor)
            rotation = output_from_anchor[:3, :3] @ rotation
            velocity = output_from_anchor[:3, :3] @ velocity
            boxes.append(
                [
                    *center,
                    *size,
                    np.arctan2(rotation[1, 0], rotation[0, 0]),
                    *velocity[:2],
                ]
            )
            labels.append(classes[near])
            ids.append(track_id)
            velocity_valid.append(bracket)
        return {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 9),
            "box_labels": np.asarray(labels, dtype=np.int64),
            "box_ids": np.asarray(ids, dtype=np.int64),
            "box_velocity_valid": np.asarray(velocity_valid, dtype=bool),
        }
