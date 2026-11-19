from dataclasses import dataclass
from typing import Any
import torch


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_tree(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


@dataclass
class BackboneState:
    kv: Any = None
    ego_tokens: Any = None
    tiny_history: Any = None
    frame_index: int = 0

    def detached(self):
        return BackboneState(detach_tree(self.kv), detach_tree(self.ego_tokens),
                             detach_tree(self.tiny_history), self.frame_index)


@dataclass
class StreamState:
    backbone: BackboneState
    tracker: Any
    gaussian_map: Any
    world_from_ego: Any = None
    timestamp: float | None = None
    signature: Any = None

    def statistics(self):
        return {"frames": self.backbone.frame_index,
                "backbone_bytes": tensor_bytes((self.backbone.kv, self.backbone.ego_tokens, self.backbone.tiny_history)),
                "tracks": len(self.tracker.tracks), **self.gaussian_map.statistics()}
