from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BackendConfig:
    name: str = "dvgt2"
    repository: str = "."
    checkpoint: str = "checkpoints/dvgt2.pt"
    window: int = 4
    metric_scale: float = 0.1
    seed: int = 17
    amp: bool = True
    tiny_dim: int = 48
    patch_size: int = 16
    finetune_last_blocks: int = 0


@dataclass
class HeadConfig:
    channels: int = 64
    attention_heads: int = 4
    semantic_classes: int = 21
    detection_classes: int = 10
    detection_queries: int = 64
    detection_layers: int = 3
    samples: int = 4
    detection_bounds: list = field(
        default_factory=lambda: [-30.0, -20.0, -3.0, 80.0, 20.0, 5.0]
    )
    occupancy_classes: int = 16
    occupancy_shape: list = field(default_factory=lambda: [80, 40, 12])
    occupancy_coarse: list = field(default_factory=lambda: [10, 5])
    occupancy_bounds: list = field(
        default_factory=lambda: [-20.0, -20.0, -3.0, 60.0, 20.0, 3.0]
    )
    occupancy_layers: int = 2
    gaussian_stride: int = 4
    sky_resolution: int = 8
    sky_class: int = 10
    # Explicit image semantic IDs: unmapped dynamic pixels must not pollute the static map.
    dynamic_classes: list = field(default_factory=list)


@dataclass
class StreamConfig:
    max_camera_skew_seconds: float = 0.03
    max_gap_seconds: float = 2.0
    score_threshold: float = 0.3
    track_max_age_seconds: float = 2.0
    track_max_count: int = 128
    association_distance: float = 4.0
    static_gaussians: int = 30000
    actor_gaussians: int = 1000
    gaussian_per_frame: int = 4000
    voxel_size: float = 0.2
    map_radius: float = 100.0
    min_opacity: float = 0.05
    min_point_confidence: float = 0.0


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    renderer: str = "gsplat"

    def validate(self):
        b, h, s = self.backend, self.heads, self.stream
        if b.name not in ("dvgt2", "tiny") or not 1 <= b.window < 24:
            raise ValueError("backend must be dvgt2/tiny and window must be in [1, 23]")
        if b.metric_scale <= 0 or b.patch_size < 1 or b.finetune_last_blocks < 0:
            raise ValueError("Invalid backbone scale, patch size, or finetuning count")
        if min(h.channels, h.attention_heads) <= 0 or h.channels % h.attention_heads:
            raise ValueError("channels must be divisible by attention_heads")
        for bounds in (h.detection_bounds, h.occupancy_bounds):
            if len(bounds) != 6 or any(bounds[i] >= bounds[i + 3] for i in range(3)):
                raise ValueError("Bounds must be [xmin,ymin,zmin,xmax,ymax,zmax]")
        if len(h.occupancy_shape) != 3 or len(h.occupancy_coarse) != 2:
            raise ValueError("occupancy_shape is XYZ; occupancy_coarse is XY")
        if (
            min(
                *h.occupancy_shape,
                *h.occupancy_coarse,
                h.gaussian_stride,
                h.sky_resolution,
                h.detection_queries,
                h.detection_layers,
                h.occupancy_layers,
            )
            <= 0
        ):
            raise ValueError("Head dimensions must be positive")
        if not -1 <= h.sky_class < h.semantic_classes or any(
            x < 0 or x >= h.semantic_classes for x in h.dynamic_classes
        ):
            raise ValueError("Semantic class mapping out of range")
        if (
            min(
                s.static_gaussians,
                s.actor_gaussians,
                s.gaussian_per_frame,
                s.track_max_count,
            )
            < 1
        ):
            raise ValueError("State budgets must be positive")
        if (
            min(
                s.voxel_size,
                s.map_radius,
                s.track_max_age_seconds,
                s.association_distance,
                s.max_gap_seconds,
            )
            <= 0
        ):
            raise ValueError("Spatial and temporal limits must be positive")
        if self.renderer not in ("reference", "gsplat"):
            raise ValueError("renderer must be reference or gsplat")
        return self

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    raw = dict(raw)
    allowed = {f.name for f in fields(Config)}
    if set(raw) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(raw) - allowed)}")
    for name, cls in (
        ("backend", BackendConfig),
        ("heads", HeadConfig),
        ("stream", StreamConfig),
    ):
        raw[name] = cls(**raw.get(name, {}))
    return Config(**raw).validate()


def load_config(path):
    with open(path, encoding="utf8") as f:
        return from_dict(yaml.safe_load(f) or {})


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
