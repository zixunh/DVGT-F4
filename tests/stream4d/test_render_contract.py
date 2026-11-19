import unittest
import torch
from stream4d.render import sample_cubemap, render_scene
from stream4d.config import from_dict
from stream4d.model import StreamingFoundation4D


class RenderContractTests(unittest.TestCase):
    def test_cubemap_faces_and_gradients(self):
        values = torch.arange(6, dtype=torch.float32) / 5
        cubemap = (
            values[:, None, None, None].expand(6, 3, 2, 2).clone().requires_grad_()
        )
        directions = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ]
        )
        colors = sample_cubemap(cubemap, directions)
        torch.testing.assert_close(colors[..., 0].flatten(), values)
        colors.sum().backward()
        self.assertTrue(torch.isfinite(cubemap.grad).all())
        self.assertTrue((cubemap.grad.flatten(1).sum(-1) > 0).all())

    def test_empty_scene_sky_and_affine(self):
        scene = dict(
            means=torch.empty(0, 3),
            scales=torch.empty(0, 3),
            quaternions=torch.empty(0, 4),
            colors=torch.empty(0, 3),
            opacities=torch.empty(0),
        )
        k = torch.tensor([[10.0, 0.0, 1.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]])
        rendered = render_scene(
            scene,
            k,
            torch.eye(4),
            (4, 4),
            backend="reference",
            sky_cubemap=torch.full((6, 3, 2, 2), 0.2),
            affine_matrix=torch.eye(3) * 2,
            affine_bias=torch.full((3,), 0.1),
        )
        torch.testing.assert_close(rendered["rgb"], torch.full((4, 4, 3), 0.5))
        self.assertEqual(float(rendered["opacity"].sum()), 0.0)

    def test_stage_switch_restores_trainability(self):
        cfg = from_dict({"backend": {"name": "tiny"}, "renderer": "reference"})
        m = StreamingFoundation4D(cfg)
        m.set_stage("gaussian")
        self.assertFalse(any(p.requires_grad for p in m.backend.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.heads.semantic.parameters()))
        m.set_stage("joint")
        self.assertTrue(all(p.requires_grad for p in m.backend.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.heads.parameters()))


if __name__ == "__main__":
    unittest.main()
