"""Budgeted static world Gaussians and actor-local Gaussians.

Voxel replacement chooses the newest observation, then applies opacity/radius budgets.
This is an online mapping baseline, not global map optimization or loop closure.
"""

import torch
from .geometry import (
    assign_points_to_boxes,
    box_poses,
    invert_pose,
    matrix_to_quaternion,
    quaternion_multiply,
)

FIELDS = ("means", "scales", "quaternions", "colors", "opacities", "normals")


def select(data, indices):
    return {k: v[indices] for k, v in data.items()}


def concat(a, b):
    return b if a is None else {k: torch.cat((a[k], b[k]), 0) for k in a}


def flatten_gaussians(data):
    return {
        k: data[k].reshape(-1, data[k].shape[-1])
        if k != "opacities"
        else data[k].reshape(-1)
        for k in FIELDS
    }


def transform_gaussians(data, transform):
    out = dict(data)
    out["means"] = data["means"] @ transform[:3, :3].T + transform[:3, 3]
    out["normals"] = data["normals"] @ transform[:3, :3].T
    out["quaternions"] = quaternion_multiply(
        matrix_to_quaternion(transform[:3, :3]), data["quaternions"]
    )
    return out


def move_actors(data, source_boxes, target_boxes, source_ids, target_ids):
    """Differentiable GT-track conditioning for held-out-time Gaussian training."""
    assignment = assign_points_to_boxes(data["means"].detach(), source_boxes)
    output = {k: v.clone() for k, v in data.items()}
    source_pose, target_pose = box_poses(source_boxes), box_poses(target_boxes)
    for i, tid in enumerate(source_ids.tolist()):
        target = (target_ids == tid).nonzero().flatten()
        mask = assignment == i
        if not mask.any():
            continue
        if not len(target):
            # Unknown target motion cannot be treated as static.
            output["opacities"][mask] = 0
            continue
        transformed = transform_gaussians(
            select(data, mask), target_pose[target[0]] @ invert_pose(source_pose[i])
        )
        for key in ("means", "normals", "quaternions"):
            output[key][mask] = transformed[key]
    return output


class GaussianMap:
    def __init__(self, cfg, dynamic_classes=()):
        self.cfg = cfg
        self.dynamic_classes = tuple(dynamic_classes)
        self.static = None
        self.actors = {}

    def compact(self, data, limit, center=None):
        valid = torch.isfinite(data["means"]).all(-1) & (
            data["opacities"] >= self.cfg.min_opacity
        )
        if center is not None:
            valid &= (data["means"] - center).norm(dim=-1) <= self.cfg.map_radius
        data = select(data, valid)
        if not len(data["means"]):
            return data
        keys = torch.floor(data["means"] / self.cfg.voxel_size).long()
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
        last = torch.full(
            (int(inverse.max()) + 1,), -1, device=keys.device, dtype=torch.long
        )
        last.scatter_reduce_(
            0, inverse, torch.arange(len(keys), device=keys.device), reduce="amax"
        )
        data = select(data, last)
        if len(data["means"]) > limit:
            indices = data["opacities"].topk(limit, sorted=False).indices
            data = select(data, indices)
        return {k: v.detach().clone() for k, v in data.items()}

    @torch.no_grad()
    def update(self, gaussians, world_from_ego, tracks):
        data = flatten_gaussians(gaussians)
        labels = gaussians["semantic_labels"].reshape(-1)
        # Select a bounded number before box association (N*K memory).
        count = min(len(labels), self.cfg.gaussian_per_frame)
        indices = data["opacities"].topk(count, sorted=False).indices
        data, labels = select(data, indices), labels[indices]
        data = transform_gaussians(data, world_from_ego)
        assignment = assign_points_to_boxes(data["means"], tracks["boxes"])
        poses = box_poses(tracks["boxes"])
        active = set(tracks["ids"].tolist())
        for tid in list(self.actors):
            if tid not in active:
                del self.actors[tid]
        for i, tid in enumerate(tracks["ids"].tolist()):
            mask = assignment == i
            if mask.any():
                local = transform_gaussians(select(data, mask), invert_pose(poses[i]))
                self.actors[tid] = self.compact(
                    concat(self.actors.get(tid), local), self.cfg.actor_gaussians
                )
        static_mask = assignment < 0
        for label in self.dynamic_classes:
            static_mask &= labels != label
        self.static = self.compact(
            concat(self.static, select(data, static_mask)),
            self.cfg.static_gaussians,
            world_from_ego[:3, 3],
        )

    def scene(self, tracks):
        data = self.static
        poses = box_poses(tracks["boxes"])
        for i, tid in enumerate(tracks["ids"].tolist()):
            if tid in self.actors:
                data = concat(data, transform_gaussians(self.actors[tid], poses[i]))
        return data

    def statistics(self):
        return dict(
            static_gaussians=0 if self.static is None else len(self.static["means"]),
            actor_gaussians=sum(len(x["means"]) for x in self.actors.values()),
            active_actor_maps=len(self.actors),
        )
