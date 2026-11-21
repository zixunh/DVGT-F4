"""PAI camera/rig geometry in meters, OpenCV RDF and vehicle FLU.

Format references: NVIDIA/ncore tools/data_converter/pai and
https://nvidia.github.io/ncore/data/sensor_models.html. The reference f-theta
polynomial is inverted numerically; the supplied inverse is only an initial guess.
"""

import io
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def pose_from_row(row):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(
        [row[k] for k in ("qx", "qy", "qz", "qw")]
    ).as_matrix()
    pose[:3, 3] = [row[k] for k in ("x", "y", "z")]
    return pose


def transform(points, pose):
    return np.einsum("...ij,...j->...i", pose[..., :3, :3], points) + pose[..., :3, 3]


class Trajectory:
    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["timestamp"])
        self.times = np.array([r["timestamp"] for r in rows], dtype=np.int64)
        if len(rows) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Egomotion timestamps must strictly increase")
        self.xyz = np.array([[r[k] for k in ("x", "y", "z")] for r in rows])
        rotations = Rotation.from_quat(
            [[r[k] for k in ("qx", "qy", "qz", "qw")] for r in rows]
        )
        self.slerp = Slerp((self.times - self.times[0]) * 1e-6, rotations)

    def at(self, times):
        times = np.asarray(times, dtype=np.int64)
        if np.any(times < self.times[0]) or np.any(times > self.times[-1]):
            raise ValueError(
                "Pose extrapolation outside egomotion support is forbidden"
            )
        flat = times.reshape(-1)
        poses = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
        poses[:, :3, :3] = self.slerp((flat - self.times[0]) * 1e-6).as_matrix()
        poses[:, :3, 3] = np.stack(
            [np.interp(flat, self.times, self.xyz[:, i]) for i in range(3)], -1
        )
        return poses.reshape(times.shape + (4, 4))


class Rectifier:
    def __init__(self, row, hw, horizontal_fov_degrees):
        import cv2

        if row["model_type"] != "ftheta":
            raise ValueError(
                "Only the verified offline PAI f-theta schema is supported"
            )
        p = json.loads(row["model_parameters"])
        if p.get("external_distortion_parameters") is not None or np.any(
            p.get("tangential_coeffs", [0, 0])
        ):
            raise ValueError(
                "External/tangential distortion requires a dedicated adapter"
            )
        h, w = hw
        self.hw = (h, w)
        self.parameters = p
        self.shutter_delay_us = int(p.get("shutter_delay_us", 0))
        f = w / (2 * np.tan(np.deg2rad(horizontal_fov_degrees) / 2))
        self.k = np.array(
            [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], dtype=np.float32
        )
        y, x = np.mgrid[:h, :w]
        rays = np.stack(
            ((x - self.k[0, 2]) / f, (y - self.k[1, 2]) / f, np.ones_like(x)), -1
        )
        uv, valid = self.project(rays)
        self.map_x, self.map_y = (
            uv[..., 0].astype(np.float32),
            uv[..., 1].astype(np.float32),
        )
        mask_bytes = row.get("ego_mask_image_png")
        if mask_bytes:
            ego = np.asarray(Image.open(io.BytesIO(mask_bytes)))
            if ego.ndim != 2 or ego.shape != (p["resolution"][1], p["resolution"][0]):
                raise ValueError("Unexpected ego mask layout")
            remapped = cv2.remap(
                ego,
                self.map_x,
                self.map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            valid &= remapped == 0
        self.valid = valid

    def project(self, rays):
        p = self.parameters
        xy = np.linalg.norm(rays[..., :2], axis=-1)
        theta = np.arctan2(xy, rays[..., 2])
        radius = np.polynomial.polynomial.polyval(theta, p["angle_to_pixeldist_poly"])
        if p["reference_poly"] == "PIXELDIST_TO_ANGLE":
            backward = np.array(p["pixeldist_to_angle_poly"])
            derivative = np.polynomial.polynomial.polyder(backward)
            for _ in range(10):
                slope = np.polynomial.polynomial.polyval(radius, derivative)
                radius -= (
                    np.polynomial.polynomial.polyval(radius, backward) - theta
                ) / np.maximum(slope, 1e-10)
            supported = theta <= p["max_angle"]
            residual = np.abs(
                np.polynomial.polynomial.polyval(radius, backward) - theta
            )
            if np.any(residual[supported] > 1e-6):
                raise ValueError("Could not invert reference camera polynomial")
        elif p["reference_poly"] != "ANGLE_TO_PIXELDIST":
            raise ValueError("Unknown reference f-theta polynomial")
        local = rays[..., :2] * (radius / np.maximum(xy, 1e-12))[..., None]
        c, d, e = p.get("linear_cde", [1, 0, 0])
        uv = local @ np.array([[c, d], [e, 1]]).T + p["principal_point"]
        width, height = p["resolution"]
        valid = (theta <= p["max_angle"]) & (radius >= 0) & np.isfinite(uv).all(-1)
        valid &= (
            (uv[..., 0] >= 0)
            & (uv[..., 0] <= width - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= height - 1)
        )
        return uv, valid

    def image(self, bgr):
        import cv2

        width, height = self.parameters["resolution"]
        if bgr.shape[:2] != (height, width):
            raise ValueError("Video resolution disagrees with camera calibration")
        image = cv2.remap(
            bgr,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return image[..., ::-1].transpose(2, 0, 1).copy()


def sparse_depth(points_ego, k, camera_to_ego, valid_image):
    """Nearest measured z at integer pixel centers; unobserved pixels stay zero."""
    camera = transform(points_ego, np.linalg.inv(camera_to_ego))
    good = np.isfinite(camera).all(-1) & (camera[:, 2] > 0.1)
    camera = camera[good]
    uv = (camera @ k.T)[:, :2] / camera[:, 2:]
    uv = np.floor(uv + 0.5).astype(np.int64)
    h, w = valid_image.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z = uv[inside], camera[inside, 2]
    depth = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(depth, uv[:, 1] * w + uv[:, 0], z)
    depth = depth.reshape(h, w)
    depth[~np.isfinite(depth) | ~valid_image] = 0
    return depth


def binary_occupancy(points, origins, bounds, shape, max_rays=2048):
    """Measured hit=1, sampled traversed free=0, everything else=-1.

    Rays end before their first measured return, are clipped to the grid, and
    hits always override free samples. No semantic classes are invented.
    """
    lo, hi = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    shape = np.asarray(shape, dtype=int)
    cell = (hi - lo) / shape
    grid = np.full(tuple(shape), -1, dtype=np.int8)
    good = np.isfinite(points).all(-1) & np.isfinite(origins).all(-1)
    points, origins = points[good], origins[good]
    if not len(points):
        return grid
    indices = np.linspace(0, len(points) - 1, min(max_rays, len(points)), dtype=int)
    end, start = points[indices], origins[indices]
    delta = end - start
    length = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(length[:, None], 1e-9)
    safe = np.where(np.abs(direction) > 1e-12, direction, 1e-12)
    a, b = (lo - start) / safe, (hi - start) / safe
    enter = np.maximum(np.minimum(a, b).max(-1), 0)
    leave = np.minimum(np.maximum(a, b).min(-1), length - np.linalg.norm(cell))
    # Sampling is deliberately conservative: not every traversed cell is marked free.
    step = float(cell.min() * 0.5)
    for offset in np.arange(
        step * 0.5, float(np.maximum(leave - enter, 0).max()), step
    ):
        distance = enter + offset
        selected = distance < leave
        xyz = start[selected] + direction[selected] * distance[selected, None]
        vox = np.floor((xyz - lo) / cell).astype(int)
        inside = ((vox >= 0) & (vox < shape)).all(-1)
        grid[tuple(vox[inside].T)] = 0
    vox = np.floor((points - lo) / cell).astype(int)
    inside = ((vox >= 0) & (vox < shape)).all(-1)
    grid[tuple(vox[inside].T)] = 1
    return grid
