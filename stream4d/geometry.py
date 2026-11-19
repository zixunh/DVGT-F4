"""Meters; ego/world FLU; cameras OpenCV RDF; quaternions wxyz.

T_a_b maps a point from frame b into frame a. Boxes are
[cx,cy,cz,dx,dy,dz,yaw,vx,vy] with yaw about the FLU +z axis.
"""
import torch
import torch.nn.functional as F


def invert_pose(t):
    r = t[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(t)
    out[..., :3, :3] = r
    out[..., :3, 3] = -(r @ t[..., :3, 3, None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def transform_points(t, p):
    return torch.einsum("...ij,...j->...i", t[..., :3, :3], p) + t[..., :3, 3]


def rdf_to_flu(like):
    return like.new_tensor([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])


def normalize_quaternion(q):
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(q.norm(dim=-1, keepdim=True) > 1e-8, F.normalize(q, dim=-1), identity)


def quaternion_to_matrix(q):
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(m):
    # Pick the largest component for stability, including rotations near pi.
    m00, m01, m02 = m[..., 0, :].unbind(-1)
    m10, m11, m12 = m[..., 1, :].unbind(-1)
    m20, m21, m22 = m[..., 2, :].unbind(-1)
    qabs = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                        1-m00+m11-m22, 1-m00-m11+m22), -1).clamp_min(0).sqrt()
    candidates = torch.stack((torch.stack((qabs[..., 0]**2, m21-m12, m02-m20, m10-m01), -1),
                              torch.stack((m21-m12, qabs[..., 1]**2, m10+m01, m02+m20), -1),
                              torch.stack((m02-m20, m10+m01, qabs[..., 2]**2, m12+m21), -1),
                              torch.stack((m10-m01, m20+m02, m21+m12, qabs[..., 3]**2), -1)), -2)
    candidates = candidates / (2*qabs[..., :, None].clamp_min(.1))
    index = qabs.argmax(-1)[..., None, None].expand(*qabs.shape[:-1], 1, 4)
    return normalize_quaternion(candidates.gather(-2, index).squeeze(-2))


def quaternion_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return normalize_quaternion(torch.stack((aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1))


def scale_intrinsics(k, old_hw, new_hw):
    # Pixel centers: resizing with align_corners=False.
    sy, sx = new_hw[0]/old_hw[0], new_hw[1]/old_hw[1]
    out = k.clone()
    out[..., 0, :] *= sx
    out[..., 1, :] *= sy
    out[..., 0, 2] += (sx-1)*.5
    out[..., 1, 2] += (sy-1)*.5
    return out


def camera_rays(k, hw, unit=False):
    h, w = hw
    y, x = torch.meshgrid(torch.arange(h, device=k.device), torch.arange(w, device=k.device), indexing="ij")
    pix = torch.stack((x, y, torch.ones_like(x)), -1).to(k.dtype)
    rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(k), pix)
    return F.normalize(rays, dim=-1) if unit else rays


def unproject_depth(depth, k, camera_to_ego):
    p = camera_rays(k, depth.shape[-2:])*depth[..., None]
    return torch.einsum("vij,vhwj->vhwi", camera_to_ego[:, :3, :3], p) + camera_to_ego[:, None, None, :3, 3]


def ego_points_to_camera(points, camera_to_ego):
    e = invert_pose(camera_to_ego)
    return torch.einsum("vij,vhwj->vhwi", e[:, :3, :3], points) + e[:, None, None, :3, 3]


def box_poses(boxes):
    t = torch.eye(4, device=boxes.device, dtype=boxes.dtype).expand(*boxes.shape[:-1], 4, 4).clone()
    c, s = boxes[..., 6].cos(), boxes[..., 6].sin()
    t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1] = c, -s, s, c
    t[..., :3, 3] = boxes[..., :3]
    return t


def transform_boxes(boxes, t):
    out = boxes.clone()
    out[..., :3] = boxes[..., :3] @ t[:3, :3].T + t[:3, 3]
    direction = torch.stack((boxes[..., 6].cos(), boxes[..., 6].sin(), torch.zeros_like(boxes[..., 6])), -1) @ t[:3, :3].T
    out[..., 6] = torch.atan2(direction[..., 1], direction[..., 0])
    velocity = F.pad(boxes[..., 7:9], (0, 1)) @ t[:3, :3].T
    out[..., 7:9] = velocity[..., :2]
    return out


def assign_points_to_boxes(points, boxes):
    """Nearest normalized box center resolves overlaps; -1 means static/unassigned."""
    if not len(boxes):
        return torch.full(points.shape[:-1], -1, device=points.device, dtype=torch.long)
    local = torch.einsum("kij,nkj->nki", invert_pose(box_poses(boxes))[:, :3, :3], points[:, None]-boxes[None, :, :3])
    norm = local.abs() / (boxes[None, :, 3:6]*.5).clamp_min(.01)
    inside = (norm <= 1).all(-1)
    scores = norm.square().sum(-1).masked_fill(~inside, float("inf"))
    indices = scores.argmin(-1)
    return torch.where(inside.any(-1), indices, -1)
