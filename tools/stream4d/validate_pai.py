"""Real PAI gradient, split-isolation and bounded-stream integration check.

This deliberately short run verifies implementation, not benchmark accuracy.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from stream4d.checkpoint import save_adapter
from stream4d.cli import check_source
from stream4d.config import load_config
from stream4d.data import iter_frames, to_device
from stream4d.losses import multitask_loss
from stream4d.model import StreamingFoundation4D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stream4d/pai.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", default="outputs/pai_validation")
    parser.add_argument("--finetune-blocks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    cfg = load_config(args.config)
    cfg.backend.finetune_last_blocks = args.finetune_blocks
    if cfg.backend.name != "dvgt2":
        raise ValueError("Real PAI validation requires the official DVGT-2 backend")
    model = StreamingFoundation4D(cfg).cuda().train()
    model.set_stage("perception")
    train_first = next(iter_frames(args.train_manifest))
    test_first = next(iter_frames(args.test_manifest))
    if (
        train_first.get("synthetic")
        or test_first.get("synthetic")
        or train_first.get("dataset") != "pai"
        or test_first.get("dataset") != "pai"
    ):
        raise ValueError("Both manifests must contain real PAI inputs")
    if (
        test_first.get("split") != "test"
        or train_first["scene_id"] == test_first["scene_id"]
    ):
        raise ValueError("Use a genuinely distinct held-out test clip")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    watched = {
        "detection": model.heads.detection.classes[-1].weight,
        "occupancy_binary": model.heads.occupancy.binary.weight,
    }
    untrained = {
        "image_semantics": model.heads.semantic.weight,
        "occupancy_semantics": model.heads.occupancy.semantic.weight,
        "bev_semantics": model.heads.occupancy.bev.weight,
    }
    before = {k: v.detach().clone() for k, v in {**watched, **untrained}.items()}
    backbone_probe = next(
        (
            p
            for n, p in model.backend.named_parameters()
            if n in model.backend.finetune_names
        ),
        None,
    )
    backbone_before = (
        backbone_probe.detach().clone() if backbone_probe is not None else None
    )
    state, scene = None, None
    details, gradients, trained_tasks = [], set(), set()
    torch.cuda.reset_peak_memory_stats()
    for frame in itertools.islice(iter_frames(args.train_manifest), args.steps):
        check_source(frame, model, SimpleNamespace(command="train"))
        if frame["scene_id"] != scene:
            scene, state = frame["scene_id"], None
        optimizer.zero_grad(set_to_none=True)
        frame = to_device(frame, "cuda")
        pred, state = model(frame, state)
        loss, terms = multitask_loss(pred, frame, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite PAI training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        gradients |= {
            n for n, p in model.backend.named_parameters() if p.grad is not None
        }
        trained_tasks |= {
            k for k, v in terms.items() if k != "total" and v.requires_grad
        }
        details.append({k: float(v.detach()) for k, v in terms.items()})
    if len(details) != args.steps:
        raise ValueError("Training manifest is shorter than --steps")
    updated = {k: not torch.equal(before[k], v) for k, v in watched.items()}
    untouched = {
        k: torch.equal(before[k], v) and v.grad is None for k, v in untrained.items()
    }
    backbone_updated = backbone_probe is not None and not torch.equal(
        backbone_before, backbone_probe
    )
    if not all(updated.values()) or not all(untouched.values()):
        raise RuntimeError(
            f"Unexpected supervised/unsupervised parameter updates: {updated}, {untouched}"
        )
    if gradients - model.backend.finetune_names or (
        args.finetune_blocks and not backbone_updated
    ):
        raise RuntimeError("Backbone finetuning/freeze contract violated")
    model.trained_tasks = sorted(trained_tasks)
    model.trained_stages = ["perception"]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    save_adapter(root / "adapter.pt", model, "perception", args.steps, False)
    training_peak = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    state, scene = None, None
    samples = []
    depth_sum, depth_count = 0.0, 0
    with torch.inference_mode():
        for frame in itertools.islice(
            iter_frames(args.test_manifest), cfg.backend.window + 4
        ):
            check_source(frame, model, SimpleNamespace(command="infer"))
            if frame["split"] != "test":
                raise ValueError("Test manifest contains non-test data")
            if frame["scene_id"] != scene:
                scene, state = frame["scene_id"], None
            frame = to_device(frame, "cuda")
            pred, state = model.step(frame, state)
            if not torch.isfinite(pred["geometry"]["points_ego"]).all():
                raise RuntimeError("Nonfinite official geometry output")
            valid = frame["depth_z"] > 0
            depth_sum += float(
                (pred["geometry"]["depth_z"][valid] - frame["depth_z"][valid])
                .abs()
                .sum()
            )
            depth_count += int(valid.sum())
            samples.append(state.statistics())
    if len(samples) <= cfg.backend.window:
        raise ValueError("Test stream must cross the cache window")
    if len({s["backbone_bytes"] for s in samples[cfg.backend.window - 1 :]}) != 1:
        raise RuntimeError("Backbone state grew after reaching the window limit")
    report = dict(
        real_data=True,
        benchmark_accuracy_claimed=False,
        official_load=model.backend.load_report,
        train_scene=train_first["scene_id"],
        test_scene=test_first["scene_id"],
        views=int(train_first["images"].shape[0]),
        input_hw=list(train_first["images"].shape[-2:]),
        train_steps=len(details),
        losses=details,
        updated_heads=updated,
        unsupervised_classifiers_unchanged=untouched,
        trained_tasks=sorted(trained_tasks),
        finetune_last_blocks=args.finetune_blocks,
        backbone_updated=backbone_updated,
        backbone_parameters_with_grad=len(gradients),
        peak_training_cuda_mb=training_peak,
        test_frames=len(samples),
        state_samples=samples,
        sparse_depth_mae_m=depth_sum / depth_count,
    )
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
