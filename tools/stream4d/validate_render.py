"""Optional CUDA integration check for gsplat RGB/radial-range and gradients."""

import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stream4d.render import render_scene


def main():
    torch.manual_seed(17)
    scene = dict(
        means=torch.tensor(
            [[0.0, 0.0, 3.0], [0.3, 0.0, 4.0]], device="cuda", requires_grad=True
        ),
        scales=torch.ones(2, 3, device="cuda") * 0.15,
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cuda"
        ),
        colors=torch.tensor([[0.7, 0.3, 0.2], [0.2, 0.7, 0.3]], device="cuda"),
        opacities=torch.ones(2, device="cuda") * 0.8,
    )
    k = torch.tensor(
        [[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]], device="cuda"
    )
    camera = torch.eye(4, device="cuda")
    outputs = {}
    for backend in ("reference", "gsplat"):
        outputs[backend] = render_scene(scene, k, camera, (32, 32), backend=backend)
        if not all(torch.isfinite(x).all() for x in outputs[backend].values()):
            raise RuntimeError(f"Nonfinite {backend} output")
    error = float(
        (outputs["reference"]["rgb"] - outputs["gsplat"]["rgb"]).abs().mean().detach()
    )
    if error > 0.02:
        raise RuntimeError(f"Reference/CUDA RGB mismatch: {error}")
    observed = outputs["gsplat"]["opacity"] > 0.05
    range_error = float(
        (outputs["reference"]["range"][observed] - outputs["gsplat"]["range"][observed])
        .abs()
        .mean()
        .detach()
    )
    if range_error > 0.05:
        raise RuntimeError(f"Radial range mismatch: {range_error}")
    outputs["gsplat"]["rgb"].sum().backward()
    if scene["means"].grad is None or not torch.isfinite(scene["means"].grad).all():
        raise RuntimeError("Invalid gsplat gradients")
    report = dict(
        cuda_forward_backward=True,
        reference_rgb_mean_abs_error=error,
        reference_camera_range_mean_abs_error=range_error,
        primitive_count=2,
        output_hw=[32, 32],
    )
    root = ROOT / "outputs/render_validation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
