"""Slow differentiable static Gaussian renderer for correctness tests / small scenes.

Reference rasterizer adapted from the sibling unofficial Foundation4D implementation.
The public renderer wrapper supplies actor transforms, sky and appearance.
This is not a replacement for the Instant NuRec production rasterizer.
"""

import torch

from .geometry import invert_pose, quaternion_to_matrix


def render_gaussians(
    means,
    scales,
    quaternions,
    colors,
    opacities,
    intrinsics,
    camera_c2w,
    height,
    width,
    chunk_size=128,
):
    if len(means) * height * width > 32_000_000:
        raise ValueError(
            "Reference renderer budget exceeded (G*H*W > 32M); downsample explicitly"
        )
    w2c = invert_pose(camera_c2w)
    camera = means @ w2c[:3, :3].T + w2c[:3, 3]
    valid = (camera[:, 2] > 0.01) & torch.isfinite(camera).all(-1)
    camera, scales, quaternions, colors, opacities = [
        v[valid] for v in (camera, scales, quaternions, colors, opacities)
    ]
    if not len(camera):
        return (
            means.sum() * 0 + means.new_zeros(height, width, 3),
            means.new_zeros(height, width),
            means.new_zeros(height, width),
        )
    order = camera[:, 2].argsort()
    camera, scales, quaternions, colors, opacities = [
        v[order] for v in (camera, scales, quaternions, colors, opacities)
    ]
    rotation = w2c[:3, :3] @ quaternion_to_matrix(quaternions)
    covariance = (rotation * scales[:, None, :].square()) @ rotation.transpose(-1, -2)
    projected = camera @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:]
    z = projected[:, 2]
    # Full perspective Jacobian, allowing skew in K.
    jacobian = (
        intrinsics[None, :2, :] * z[:, None, None]
        - projected[:, :2, None] * intrinsics[None, 2:3, :]
    ) / z[:, None, None].square()
    cov2 = (
        jacobian @ covariance @ jacobian.transpose(-1, -2)
        + 0.3 * torch.eye(2, device=means.device)[None]
    )
    precision = torch.linalg.inv(cov2)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=means.device),
        torch.arange(width, device=means.device),
        indexing="ij",
    )
    pixel = torch.stack((xx, yy), -1).to(means.dtype)
    transmittance = means.new_ones(height, width)
    rgb, distance = means.new_zeros(height, width, 3), means.new_zeros(height, width)
    for start in range(0, len(camera), chunk_size):
        sl = slice(start, start + chunk_size)
        delta = pixel[None] - uv[sl, None, None]
        power = torch.einsum("nhwi,nij,nhwj->nhw", delta, precision[sl], delta)
        alpha = (opacities[sl, None, None] * (-0.5 * power).exp()).clamp(max=0.99)
        cumulative = torch.cumprod(1 - alpha, dim=0)
        before = torch.cat((torch.ones_like(cumulative[:1]), cumulative[:-1]), 0)
        weight = transmittance[None] * before * alpha
        rgb = rgb + torch.einsum("nhw,nc->hwc", weight, colors[sl])
        distance = distance + (weight * camera[sl].norm(dim=-1)[:, None, None]).sum(0)
        transmittance = transmittance * cumulative[-1]
    opacity = 1 - transmittance
    return rgb, opacity, distance / opacity.clamp_min(1e-6)
