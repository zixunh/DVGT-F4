from types import SimpleNamespace
import unittest
import torch

from stream4d.cli import check_source
from stream4d.config import load_config
from stream4d.losses import binary_occupancy_loss, detection_loss, multitask_loss
from stream4d.pai.annotations import DETECTION_CLASSES


class SparseSupervisionTests(unittest.TestCase):
    def test_frozen_depth_does_not_create_fake_semantic_training_path(self):
        semantic = torch.zeros(1, 2, 2, 2, requires_grad=True)
        pred = dict(
            semantic_logits=semantic, geometry=dict(depth_z=torch.ones(1, 2, 2))
        )
        loss, _ = multitask_loss(pred, dict(depth_z=torch.full((1, 2, 2), 2.0)), None)
        self.assertFalse(loss.requires_grad)

    def test_unknown_occupancy_has_zero_gradient_and_semantics_untrained(self):
        logits = torch.zeros(3, requires_grad=True)
        semantic = torch.randn(3, 4, requires_grad=True)
        loss = binary_occupancy_loss(
            [dict(binary_logits=logits, semantic_logits=semantic)],
            torch.tensor([-1, 0, 1]),
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0)
        self.assertGreater(float(logits.grad[1]), 0)
        self.assertLess(float(logits.grad[2]), 0)
        self.assertIsNone(semantic.grad)

    def test_missing_velocity_is_masked_in_matching_and_regression(self):
        boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        logits = torch.zeros(1, 1)
        encoded = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, -20.0]], requires_grad=True
        )
        layers = [dict(class_logits=logits, encoded_boxes=encoded)]
        labels = torch.zeros(1, dtype=torch.long)
        masked = detection_loss(layers, boxes, labels, torch.tensor([False]))
        masked.backward()
        self.assertTrue((encoded.grad[0, 8:] == 0).all())
        observed = detection_loss(layers, boxes, labels, torch.tensor([True]))
        self.assertGreater(float(observed.detach()), float(masked.detach()))

    def test_test_split_cannot_be_used_for_training(self):
        cfg = load_config("configs/stream4d/pai.yaml")
        model = SimpleNamespace(config=cfg)
        frame = dict(
            dataset="pai",
            detection_classes=DETECTION_CLASSES,
            occupancy_bounds=cfg.heads.occupancy_bounds,
            split="test",
            training_eligible=False,
            foundation4d_eval=False,
        )
        with self.assertRaisesRegex(ValueError, "eligible train"):
            check_source(frame, model, SimpleNamespace(command="train"))
        check_source(frame, model, SimpleNamespace(command="infer"))
        frame.update(split="train", training_eligible=True, foundation4d_eval=True)
        with self.assertRaisesRegex(ValueError, "eligible train"):
            check_source(frame, model, SimpleNamespace(command="train"))


if __name__ == "__main__":
    unittest.main()
