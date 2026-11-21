"""Masked training objectives; missing labels never become negative supervision."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .gaussian_map import flatten_gaussians, transform_gaussians, move_actors
from .geometry import transform_boxes
from .render import render_scene


def semantic_loss(logits, labels):
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0
    if (labels[valid] >= logits.shape[-1]).any():
        raise ValueError(
            "Semantic class ID out of range; provide an explicit dataset remapping"
        )
    x, y = logits[valid], labels[valid].long()
    ce = F.cross_entropy(x, y)
    probability = x.softmax(-1)
    target = F.one_hot(y, x.shape[-1]).float()
    present = target.sum(0) > 0
    dice = 1 - (2 * (probability * target).sum(0) + 1) / (
        probability.sum(0) + target.sum(0) + 1
    )
    return ce + dice[present].mean()


def focal(logits, targets):
    p = logits.sigmoid()
    pt = p * targets + (1 - p) * (1 - targets)
    return (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        * (1 - pt) ** 2
        * (0.25 * targets + 0.75 * (1 - targets))
    )


def detection_loss(layers, boxes, labels, velocity_valid=None):
    valid = labels >= 0
    if velocity_valid is None:
        velocity_valid = torch.ones_like(labels, dtype=torch.bool)
    if velocity_valid.shape != labels.shape:
        raise ValueError("box_velocity_valid must match box_labels")
    velocity_valid = velocity_valid[valid].bool()
    boxes, labels = boxes[valid], labels[valid].long()
    if len(boxes) and ((boxes[:, 3:6] <= 0).any() or not torch.isfinite(boxes).all()):
        raise ValueError("Invalid detection annotations")
    encoded = torch.cat(
        (
            boxes[:, :3],
            boxes[:, 3:6].log(),
            boxes[:, 6:7].sin(),
            boxes[:, 6:7].cos(),
            boxes[:, 7:9],
        ),
        -1,
    )
    terms = []
    for layer in layers:
        logits, pred = layer["class_logits"], layer["encoded_boxes"]
        if len(labels) and labels.max() >= logits.shape[-1]:
            raise ValueError("Detection label out of range")
        target = torch.zeros_like(logits)
        regression = pred.sum() * 0
        weight = (
            pred.new_tensor([1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0.2])
            .expand(len(labels), -1)
            .clone()
        )
        weight[:, 8:] *= velocity_valid[:, None]
        if len(labels):
            with torch.no_grad():
                p = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
                cost_cls = -0.25 * (1 - p) ** 2 * p.log() + 0.75 * p**2 * (1 - p).log()
                distance = ((pred[:, None] - encoded[None]).abs() * weight[None]).sum(
                    -1
                )
                cost = 2 * cost_cls[:, labels] + 0.25 * distance
                row, col = linear_sum_assignment(cost.cpu().numpy())
                row = torch.as_tensor(row, device=pred.device)
                col = torch.as_tensor(col, device=pred.device)
            target[row, labels[col]] = 1
            regression = ((pred[row] - encoded[col]).abs() * weight[col]).sum() / max(
                len(labels), 1
            )
        terms.append(
            2 * focal(logits, target).sum() / max(len(labels), 1) + 0.25 * regression
        )
    return torch.stack(terms).mean()


def binary_occupancy_loss(layers, labels):
    """Sparse LiDAR hit/free supervision, with no gradient to semantic classifiers."""
    if ((labels < -1) | (labels > 1)).any():
        raise ValueError("Binary occupancy uses -1 unknown, 0 free, 1 occupied")
    terms = []
    for layer in layers:
        logits = layer["binary_logits"]
        if labels.shape != logits.shape:
            raise ValueError(
                "Binary occupancy grid must match the configured XYZ shape"
            )
        classes = []
        for value in (0, 1):
            mask = labels == value
            if mask.any():
                classes.append(
                    F.binary_cross_entropy_with_logits(
                        logits[mask], torch.full_like(logits[mask], value)
                    )
                )
        terms.append(torch.stack(classes).mean() if classes else logits.sum() * 0)
    return torch.stack(terms).mean()


def occupancy_loss(layers, labels):
    valid = labels >= 0
    terms = []
    for pred in layers:
        if labels.shape != pred["binary_logits"].shape:
            raise ValueError(
                "Occupancy label grid must match configured XYZ shape and bounds"
            )
        if not valid.any():
            terms.append(pred["binary_logits"].sum() * 0)
            continue
        occupied = labels > 0
        bce = F.binary_cross_entropy_with_logits(
            pred["binary_logits"][valid], occupied[valid].float()
        )
        sem = torch.where(occupied, labels - 1, -1)
        semloss = semantic_loss(pred["semantic_logits"], sem)
        height = torch.arange(labels.shape[-1], device=labels.device)
        top = torch.where(occupied, height, -1).amax(-1)
        bev = sem.gather(-1, top.clamp_min(0)[..., None]).squeeze(-1)
        bev = torch.where(top >= 0, bev, -1)
        terms.append(bce + semloss + 0.5 * semantic_loss(pred["bev_logits"], bev))
    return torch.stack(terms).mean()


def reconstruction_loss(pred, frame, renderer, perceptual=None):
    required = (
        "target_rgb",
        "target_intrinsics",
        "target_camera_to_world",
        "world_from_ego",
    )
    if any(k not in frame for k in required):
        raise ValueError(
            "Gaussian training requires held-out target RGB/calibration and source GT world_from_ego"
        )
    if not frame.get("target_is_held_out", False):
        raise ValueError(
            "Explicit target_is_held_out=true is required for reconstruction training"
        )
    gs = pred["gaussians"]
    source_pose = frame["world_from_ego"]
    data = transform_gaussians(flatten_gaussians(gs), source_pose)
    if "boxes" in frame and len(frame["boxes"]):
        for key in ("box_ids", "target_boxes_world", "target_box_ids"):
            if key not in frame:
                raise ValueError(f"Dynamic reconstruction requires {key}")
        data = move_actors(
            data,
            transform_boxes(frame["boxes"], source_pose),
            frame["target_boxes_world"],
            frame["box_ids"],
            frame["target_box_ids"],
        )
    terms = []
    for v, rgb_gt in enumerate(frame["target_rgb"]):
        h, w = rgb_gt.shape[-2:]
        source_view = int(
            frame.get("target_camera_indices", list(range(len(frame["target_rgb"]))))[v]
        )
        out = render_scene(
            data,
            frame["target_intrinsics"][v],
            frame["target_camera_to_world"][v],
            (h, w),
            backend=renderer,
            sky_cubemap=gs["sky_cubemap"],
            sky_to_world=source_pose,
            affine_matrix=gs["affine_matrix"][source_view],
            affine_bias=gs["affine_bias"][source_view],
        )
        target = rgb_gt.movedim(0, -1)
        valid = frame.get(
            "target_valid", torch.ones_like(frame["target_rgb"][:, 0], dtype=torch.bool)
        )[v].bool()
        valid = valid & torch.isfinite(target).all(-1)
        if not valid.any():
            continue
        loss = (out["rgb"][valid] - target[valid]).square().mean()
        if perceptual is not None:
            masked = torch.where(valid[..., None], out["rgb"], target)
            loss = (
                loss
                + 0.2
                * perceptual(
                    masked.movedim(-1, 0)[None] * 2 - 1, rgb_gt[None] * 2 - 1
                ).mean()
            )
        if "target_non_sky" in frame:
            non_sky = frame["target_non_sky"][v].float()
            loss = loss + 2 * (out["opacity"][valid] - non_sky[valid]).square().mean()
            sky_mask = valid & (non_sky < 0.5)
            if sky_mask.any():
                loss = (
                    loss
                    + 0.01 * (out["sky_rgb"][sky_mask] - target[sky_mask]).abs().mean()
                )
        if "target_range" in frame:
            gt = frame["target_range"][v]
            dm = valid & torch.isfinite(gt) & (gt > 0)
            if dm.any():
                loss = (
                    loss
                    + 0.1
                    * (
                        out["range"][dm].clamp_min(0.1).reciprocal()
                        - gt[dm].reciprocal()
                    )
                    .square()
                    .mean()
                )
        terms.append(loss)
    if not terms:
        raise ValueError("No valid held-out render pixels")
    source_rgb = F.interpolate(
        frame["images"], gs["depth"].shape[-2:], mode="bilinear", align_corners=False
    ).movedim(1, -1)
    total = torch.stack(terms).mean() + 0.01 * (gs["colors"] - source_rgb).abs().mean()
    total = (
        total
        + 0.01
        * ((gs["affine_matrix"] - torch.eye(3, device=source_pose.device)) ** 2).mean()
        + 0.01 * gs["affine_bias"].square().mean()
    )
    if "normals_ego" in frame:
        normals = F.interpolate(
            frame["normals_ego"].movedim(-1, 1),
            gs["depth"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)
        mask = torch.isfinite(normals).all(-1) & (normals.norm(dim=-1) > 0.5)
        if mask.any():
            total = (
                total
                + 0.2
                * (
                    1
                    - (F.normalize(normals[mask], dim=-1) * gs["normals"][mask]).sum(-1)
                ).mean()
            )
    return total


def multitask_loss(pred, frame, config, stage="perception", perceptual=None):
    losses = {}
    if stage != "gaussian":
        if "semantics" in frame:
            losses["semantic"] = semantic_loss(
                pred["semantic_logits"].movedim(1, -1), frame["semantics"]
            )
        if "boxes" in frame and "box_labels" in frame:
            losses["detection"] = 0.1 * detection_loss(
                pred["detections"],
                frame["boxes"],
                frame["box_labels"],
                frame.get("box_velocity_valid"),
            )
        if "occupancy" in frame:
            losses["occupancy"] = 0.2 * occupancy_loss(
                pred["occupancy"], frame["occupancy"]
            )
        elif "occupancy_binary" in frame:
            losses["occupancy_binary"] = 0.2 * binary_occupancy_loss(
                pred["occupancy"], frame["occupancy_binary"]
            )
        if "depth_z" in frame:
            gt = frame["depth_z"]
            p = pred["geometry"]["depth_z"]
            valid = torch.isfinite(gt) & (gt > 0)
            if valid.any():
                losses["depth"] = (p[valid] - gt[valid]).abs().mean() / 40
        if "points_ego" in frame:
            gt = frame["points_ego"]
            p = pred["geometry"]["points_ego"]
            valid = torch.isfinite(gt).all(-1)
            if valid.any():
                losses["points"] = (p[valid] - gt[valid]).abs().mean() / 40
    if stage in ("gaussian", "joint"):
        losses["gaussian"] = 4 * reconstruction_loss(
            pred, frame, config.renderer, perceptual
        )
    if not losses:
        raise ValueError("No applicable supervision was provided")
    # An unrelated zero-valued graph creates zero gradients and AdamW decay on
    # unsupervised heads, and hides frozen-geometry-only training mistakes.
    total = torch.stack(list(losses.values())).sum()
    return total, {**losses, "total": total}
