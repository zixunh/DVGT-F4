"""Explicit optional integration check: real official weights, synthetic input.

This is an interface/caching/gradient check and never reports benchmark accuracy.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from stream4d.config import load_config
from stream4d.data import iter_frames, make_synthetic, to_device
from stream4d.model import StreamingFoundation4D
from stream4d.losses import multitask_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/stream4d/dvgt2.yaml"))
    p.add_argument("--output", default=str(ROOT / "outputs/pretrained_validation"))
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--train-step", action="store_true")
    p.add_argument("--finetune-blocks", type=int, default=0)
    p.add_argument("--size", nargs=2, type=int, default=[64, 96])
    p.add_argument("--views", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    cfg = load_config(args.config)
    cfg.backend.finetune_last_blocks = args.finetune_blocks
    if cfg.backend.name != "dvgt2":
        raise ValueError("This check requires the real DVGT-2 backend")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_synthetic(
        output / "inputs", cfg, args.frames, args.views, tuple(args.size)
    )
    model = StreamingFoundation4D(cfg).cuda().eval()
    samples = []
    state = None
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for frame in iter_frames(manifest):
            result, state = model.step(to_device(frame, "cuda"), state)
            if not torch.isfinite(result["geometry"]["points_ego"]).all():
                raise RuntimeError("Nonfinite point output")
            samples.append(state.statistics())
    stable = [x["backbone_bytes"] for x in samples[cfg.backend.window :]]
    if stable and len(set(stable)) != 1:
        raise RuntimeError("Backbone state grew after window filled")
    record = dict(
        official_load=model.backend.load_report,
        input_hw=args.size,
        views=args.views,
        frames=args.frames,
        elapsed_seconds=time.monotonic() - start,
        peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20,
        state_samples=samples,
        synthetic_input=True,
        new_heads_random=True,
    )
    if args.train_step:
        # Fresh state: inference tensors must not enter the autograd validation.
        model.train()
        model.set_stage("perception")
        optimizer = torch.optim.AdamW(
            [x for x in model.parameters() if x.requires_grad], lr=1e-4
        )
        first = to_device(next(iter_frames(manifest)), "cuda")
        before = model.heads.semantic.weight.detach().clone()
        pred, _ = model(first)
        loss, _ = multitask_loss(pred, first, cfg, "perception")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [x for x in model.parameters() if x.requires_grad],
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if torch.equal(before, model.heads.semantic.weight):
            raise RuntimeError("New head did not update")
        gradients = {
            name for name, x in model.backend.named_parameters() if x.grad is not None
        }
        if gradients - model.backend.finetune_names:
            raise RuntimeError("Frozen backbone received gradients")
        if args.finetune_blocks and not gradients:
            raise RuntimeError("Requested finetuned blocks received no gradients")
        record["training_step"] = {
            "loss": float(loss.detach()),
            "head_updated": True,
            "finetune_blocks": args.finetune_blocks,
            "backbone_parameters_with_grad": len(gradients),
            "peak_cuda_mb_including_training": torch.cuda.max_memory_allocated()
            / 2**20,
        }
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
