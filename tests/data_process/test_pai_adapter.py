import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from stream4d.pai.annotations import Tracks
from stream4d.pai.geometry import (
    Rectifier,
    Trajectory,
    binary_occupancy,
    sparse_depth,
    transform,
)
from stream4d.pai.io import StoredMember


def motion_row(t, x):
    return dict(timestamp=t, x=x, y=0, z=0, qx=0, qy=0, qz=0, qw=1)


class PAIAdapterTests(unittest.TestCase):
    def test_stored_zip_reads_are_bounded_and_seekable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("a", b"abcdef")
                z.writestr("b", b"must not leak")
            with zipfile.ZipFile(path) as z:
                info = z.getinfo("a")
            with StoredMember(path, info) as member:
                self.assertEqual(member.read(100), b"abcdef")
                self.assertEqual(member.read(), b"")
                member.seek(-2, 2)
                self.assertEqual(member.read(), b"ef")
                member.seek(99)
                self.assertEqual(member.read(), b"")
                with self.assertRaises(ValueError):
                    member.seek(-1)

    def test_ftheta_inverse_uses_reference_and_linear_term(self):
        params = dict(
            resolution=[100, 80],
            principal_point=[49, 39],
            reference_poly="PIXELDIST_TO_ANGLE",
            pixeldist_to_angle_poly=[0, 0.02, 0, 1e-6, 0, 0],
            angle_to_pixeldist_poly=[0, 40, 0, 0, 0, 0],
            max_angle=1.4,
            linear_cde=[1.1, 0.02, -0.01],
        )
        rect = Rectifier(
            dict(model_type="ftheta", model_parameters=json.dumps(params)), (16, 32), 70
        )
        radius = 10
        theta = 0.02 * radius + 1e-6 * radius**3
        uv, valid = rect.project(
            np.array([[0, 0, 1], [np.sin(theta), 0, np.cos(theta)]])
        )
        np.testing.assert_allclose(uv, [[49, 39], [60, 38.9]], atol=1e-7)
        self.assertTrue(valid.all())

    def test_motion_compensation_and_no_extrapolation(self):
        trajectory = Trajectory([motion_row(0, 0), motion_row(2000000, 2)])
        anchor = transform(np.array([9, 0, 0]), trajectory.at(1000000))
        current = transform(anchor, np.linalg.inv(trajectory.at(2000000)))
        np.testing.assert_allclose(current, [8, 0, 0])
        with self.assertRaises(ValueError):
            trajectory.at(-1)

    def test_static_world_object_has_zero_absolute_velocity(self):
        trajectory = Trajectory([motion_row(0, 0), motion_row(200000, 2)])
        rows = []
        for time, rig_x in [(0, 10), (200000, 8)]:
            rows.append(
                dict(
                    timestamp_us=time,
                    reference_frame_timestamp_us=time,
                    reference_frame="rig",
                    track_id="stationary",
                    label_class="automobile",
                    center_x=rig_x,
                    center_y=0,
                    center_z=1,
                    size_x=4,
                    size_y=2,
                    size_z=2,
                    orientation_x=0,
                    orientation_y=0,
                    orientation_z=0,
                    orientation_w=1,
                )
            )
        tracks = Tracks(rows, trajectory)
        result = tracks.at(100000, np.linalg.inv(trajectory.at(100000)))
        np.testing.assert_allclose(result["boxes"][0, [0, 7, 8]], [9, 0, 0], atol=1e-6)
        self.assertTrue(result["box_velocity_valid"][0])
        self.assertEqual(len(tracks.at(1000000, np.eye(4))["boxes"]), 0)

    def test_zbuffer_keeps_nearest_depth_and_ignores_behind_camera(self):
        points = np.array([[0, 0, 4], [0, 0, 8], [0, 0, -2]])
        k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        mask = np.ones((3, 3), dtype=bool)
        z = sparse_depth(points, k, np.eye(4), mask)
        self.assertEqual(z[1, 1], 4)
        self.assertEqual(np.count_nonzero(z), 1)
        mask[1, 1] = False
        self.assertEqual(sparse_depth(points, k, np.eye(4), mask).sum(), 0)

    def test_occupancy_does_not_mark_unseen_or_behind_return_free(self):
        points = np.array([[5.5, 0.5, 0.5]])
        origins = np.array([[0.5, 0.5, 0.5]])
        grid = binary_occupancy(points, origins, [0, 0, 0, 8, 2, 2], [8, 2, 2])
        self.assertEqual(grid[0, 0, 0], 0)
        self.assertEqual(grid[5, 0, 0], 1)
        self.assertEqual(grid[7, 0, 0], -1)
        self.assertTrue((grid[:, 1, 1] == -1).all())


if __name__ == "__main__":
    unittest.main()
