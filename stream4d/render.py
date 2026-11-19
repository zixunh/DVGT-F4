"""Reference and CUDA gsplat rasterization with radial range, sky and appearance."""

import torch
import torch.nn.functional as F
from .geometry import invert_pose, camera_rays
from ._reference_render import render_gaussians


def sample_cubemap(cubemap, directions):
    d = F.normalize(directions, dim=-1)
    major = d.abs().argmax(-1)
    axis = d.gather(-1, major[..., None]).squeeze(-1)
    face = 2 * major + (axis < 0).long()
    x, y, z = d.unbind(-1)
    denom = axis.abs().clamp_min(1e-6)
    uv = (
        torch.stack(
            (
                torch.stack((-z, -y), -1),
                torch.stack((z, -y), -1),
                torch.stack((x, z), -1),
                torch.stack((x, -z), -1),
                torch.stack((x, -y), -1),
                torch.stack((-x, -y), -1),
            ),
            0,
        )
        / denom[None, ..., None]
    )
    values = F.grid_sample(
        cubemap, uv.clamp(-1, 1), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    return values.gather(0, face[None, ..., None].expand(1, *face.shape, 3))[0]


def rasterize(scene, k, camera_to_world, hw, backend="gsplat"):
    h, w = hw
    if backend == "reference":
        return render_gaussians(
            scene["means"],
            scene["scales"],
            scene["quaternions"],
            scene["colors"],
            scene["opacities"],
            k,
            camera_to_world,
            h,
            w,
        )
    if backend != "gsplat":
        raise ValueError("Unknown rasterization backend")
    if not scene["means"].is_cuda:
        raise ValueError("gsplat backend requires CUDA tensors")
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError("Install the render extra to use gsplat") from exc
    if not len(scene["means"]):
        zero = scene["means"].sum() * 0
        return (
            zero + k.new_zeros(h, w, 3),
            zero + k.new_zeros(h, w),
            zero + k.new_zeros(h, w),
        )
    w2c = invert_pose(camera_to_world)
    camera = scene["means"] @ w2c[:3, :3].T + w2c[:3, 3]
    # Render range as an extra feature channel, avoiding gsplat's z-depth/range ambiguity.
    features = torch.cat((scene["colors"], camera.norm(dim=-1, keepdim=True)), -1)
    # gsplat samples pixel centers at (j+.5,i+.5); our OpenCV K/rays use
    # integer pixel centers. Shift only the rasterizer's K to keep both consistent.
    raster_k = k.clone()
    raster_k[0, 2] += 0.5
    raster_k[1, 2] += 0.5
    rendered, alpha, _ = rasterization(
        means=scene["means"],
        quats=scene["quaternions"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        colors=features,
        viewmats=w2c[None],
        Ks=raster_k[None],
        width=w,
        height=h,
        render_mode="RGB",
        packed=True,
    )
    alpha = alpha[0, ..., 0]
    return rendered[0, ..., :3], alpha, rendered[0, ..., 3] / alpha.clamp_min(1e-6)


def render_scene(
    scene,
    k,
    camera_to_world,
    hw,
    *,
    backend="gsplat",
    sky_cubemap=None,
    sky_to_world=None,
    affine_matrix=None,
    affine_bias=None,
):
    rgb, alpha, distance = rasterize(scene, k, camera_to_world, hw, backend)
    sky = torch.zeros_like(rgb)
    if sky_cubemap is not None:
        rays = camera_rays(k[None], hw, unit=True)[0] @ camera_to_world[:3, :3].T
        if sky_to_world is not None:
            rays = rays @ sky_to_world[:3, :3]
        sky = sample_cubemap(sky_cubemap, rays)
    rgb = rgb + (1 - alpha)[..., None] * sky
    if affine_matrix is not None:
        rgb = rgb @ affine_matrix.T
    if affine_bias is not None:
        rgb = rgb + affine_bias
    return dict(rgb=rgb.clamp(0, 1), opacity=alpha, range=distance, sky_rgb=sky)
