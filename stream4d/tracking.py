"""Causal 3D tracking: planar constant-velocity Kalman + class-gated Hungarian.

Unlike Foundation4D's offline two-pass RTS smoother, no future observation is used.
"""

from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    label: int
    box: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    last_seen: float
    score: float
    hits: int = 1


class OnlineTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1
        self.timestamp = None

    def update(self, boxes, labels, scores, timestamp):
        boxes = boxes.detach().float().cpu().numpy()
        labels = labels.detach().long().cpu().numpy()
        scores = scores.detach().float().cpu().numpy()
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("Tracker timestamps must strictly increase")
        dt = 0 if self.timestamp is None else timestamp - self.timestamp
        self.timestamp = timestamp
        for tid in list(self.tracks):
            if timestamp - self.tracks[tid].last_seen > self.cfg.track_max_age_seconds:
                del self.tracks[tid]
        ids = list(self.tracks)
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        q = np.diag([0.1, 0.1, 1.0, 1.0]) * max(dt, 1e-3)
        for tr in self.tracks.values():
            tr.mean = f @ tr.mean
            tr.covariance = f @ tr.covariance @ f.T + q
            tr.box[:2], tr.box[7:9] = tr.mean[:2], tr.mean[2:]
        assigned, matched = np.full(len(boxes), -1, dtype=np.int64), set()
        if ids and len(boxes):
            centers = np.stack([self.tracks[i].mean[:2] for i in ids])
            dist = np.linalg.norm(centers[:, None] - boxes[None, :, :2], axis=-1)
            valid = (dist <= self.cfg.association_distance) & (
                np.array([self.tracks[i].label for i in ids])[:, None] == labels[None]
            )
            rows, cols = linear_sum_assignment(np.where(valid, dist, 1e6))
            for r, c in zip(rows, cols):
                if not valid[r, c]:
                    continue
                tr = self.tracks[ids[r]]
                observation = np.r_[boxes[c, :2], boxes[c, 7:9]]
                noise = np.diag([0.5, 0.5, 2.0, 2.0])
                gain = np.linalg.solve((tr.covariance + noise).T, tr.covariance.T).T
                tr.mean += gain @ (observation - tr.mean)
                ident = np.eye(4) - gain
                tr.covariance = ident @ tr.covariance @ ident.T + gain @ noise @ gain.T
                # Circular yaw smoothing does not jump across +/-pi.
                yaw_delta = np.arctan2(
                    np.sin(boxes[c, 6] - tr.box[6]), np.cos(boxes[c, 6] - tr.box[6])
                )
                yaw = tr.box[6] + 0.7 * yaw_delta
                tr.box = boxes[c].copy()
                tr.box[6] = yaw
                tr.box[:2], tr.box[7:9] = tr.mean[:2], tr.mean[2:]
                tr.last_seen, tr.score = timestamp, float(scores[c])
                tr.hits += 1
                assigned[c] = tr.id
                matched.add(c)
        for c in np.argsort(-scores):
            if c in matched or len(self.tracks) >= self.cfg.track_max_count:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                tid,
                int(labels[c]),
                boxes[c].copy(),
                np.r_[boxes[c, :2], boxes[c, 7:9]].astype(np.float64),
                np.eye(4),
                timestamp,
                float(scores[c]),
            )
            assigned[c] = tid
        return assigned

    def tensors(self, device):
        tracks = sorted(self.tracks.values(), key=lambda x: x.id)
        return dict(
            ids=torch.tensor([x.id for x in tracks], device=device, dtype=torch.long),
            labels=torch.tensor(
                [x.label for x in tracks], device=device, dtype=torch.long
            ),
            scores=torch.tensor(
                [x.score for x in tracks], device=device, dtype=torch.float32
            ),
            boxes=torch.as_tensor(
                np.stack([x.box for x in tracks]) if tracks else np.empty((0, 9)),
                device=device,
                dtype=torch.float32,
            ),
            observed=torch.tensor(
                [x.last_seen == self.timestamp for x in tracks],
                device=device,
                dtype=torch.bool,
            ),
        )
