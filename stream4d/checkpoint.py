from pathlib import Path
import torch
from .config import from_dict
from .model import StreamingFoundation4D


def save_adapter(path, model, stage, steps, synthetic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.config.backend.name == "tiny":
        backend = model.backend.state_dict()
    else:
        names = model.backend.finetune_names
        backend = {k: v for k, v in model.backend.state_dict().items() if k in names}
    payload = dict(
        format="dvgt-f4-adapter-v1",
        config=model.config.to_dict(),
        heads=model.heads.state_dict(),
        backend=backend,
        trained_stages=sorted(set(getattr(model, "trained_stages", [])) | {stage}),
        steps=steps,
        synthetic=synthetic,
    )
    torch.save(payload, path)


def load_adapter(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "dvgt-f4-adapter-v1":
        raise ValueError("Unsupported adapter checkpoint")
    model = StreamingFoundation4D(from_dict(saved["config"]))
    model.heads.load_state_dict(saved["heads"], strict=True)
    if model.config.backend.name == "tiny":
        model.backend.load_state_dict(saved["backend"], strict=True)
    else:
        expected = model.backend.finetune_names
        if set(saved["backend"]) != expected:
            raise ValueError(
                "Finetuned backbone keys do not match configured trainable blocks"
            )
        report = model.backend.load_state_dict(saved["backend"], strict=False)
        if report.unexpected_keys:
            raise ValueError(f"Unexpected backbone keys: {report.unexpected_keys}")
    model.trained_stages = saved["trained_stages"]
    model.synthetic_checkpoint = bool(saved["synthetic"])
    return model.to(device), saved
