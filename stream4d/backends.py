"""Original DVGT-2 modules with an external, bounded, per-stream cache.

The tiny backend is only a trainable integration fixture; never a DVGT-2 substitute.
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F

from .config import project_path
from .geometry import rdf_to_flu, unproject_depth
from .state import BackboneState


def trim_kv(kv, keep_frames, tokens_per_frame):
    if kv is None or keep_frames == 0:
        return None
    count = keep_frames*tokens_per_frame
    # clone releases backing storage belonging to discarded frames.
    return [(k[..., -count:, :].detach().clone(), v[..., -count:, :].detach().clone()) for k, v in kv]


class DVGT2Backend(nn.Module):
    feature_dim = 3072
    patch_size = 16
    pretrained = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        root, checkpoint = project_path(cfg.repository).resolve(), project_path(cfg.checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Official checkpoint missing: {checkpoint}. Run tools/stream4d/setup.py --weights")
        if not (root/'third_party/dinov3/hubconf.py').is_file():
            raise FileNotFoundError("DINOv3 code missing; run tools/stream4d/setup.py")
        anchors = root/'data_annotation/anchors/train_total_20'
        for name in ('future_anchors_20_rdf.npz', 'past_anchors_20_rdf.npz'):
            if not (anchors/name).is_file():
                raise FileNotFoundError(f"Official anchor missing: {anchors/name}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from dvgt.models.architectures.dvgt2 import DVGT2
        dino_root = str(root/'third_party/dinov3')
        if dino_root not in sys.path:
            sys.path.insert(0, dino_root)
        from dinov3.hub.backbones import dinov3_vitl16
        encoder = dinov3_vitl16(pretrained=False)

        pose_conf = dict(_target_='dvgt.models.heads.dvgt2_ego_pose_head.DVGT2EgoPoseHead',
                         future_frame_window=8, relative_pose_window=1, gt_scale_factor=cfg.metric_scale,
                         enable_ego_status=False, traj_anchor_filepath=str(anchors/'future_anchors_20_rdf.npz'),
                         pose_translation_anchor_filepath=str(anchors/'past_anchors_20_rdf.npz'))
        # Keep constructors on CPU: DINO has nonpersistent buffers that meta construction would lose.
        self.core = DVGT2(dino_v3_weight_path=None, ego_pose_head_conf=pose_conf,
                          use_causal_mask=True, patch_embed_module=encoder)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        weights = saved.get('model', saved)
        result = self.core.load_state_dict(weights, strict=True, assign=True)
        self.load_report = {'checkpoint': str(checkpoint), 'keys': len(weights),
                            'parameters': sum(p.numel() for p in self.core.parameters()),
                            'missing_keys': list(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys)}
        self.core.requires_grad_(False)
        n = cfg.finetune_last_blocks
        if n:
            if n > len(self.core.aggregator.cross_frame_blocks):
                raise ValueError('finetune_last_blocks exceeds upstream depth')
            for name in ('intra_view_blocks', 'cross_view_blocks', 'cross_frame_blocks'):
                for block in getattr(self.core.aggregator, name)[-n:]:
                    block.requires_grad_(True)
        self.core.eval()

    def train(self, mode=True):
        super().train(mode)
        # Upstream's training branch disables KV cache. Streaming fine-tuning uses
        # eval attention with gradients and detached historical state (one-step TBPTT).
        # The original diffusion pose decoder remains frozen/no-grad.
        self.core.eval()
        return self

    def step(self, images, intrinsics, extrinsics, state):
        del intrinsics, extrinsics  # Official DVGT-2 is calibration-free; added heads use calibration.
        from dvgt.utils.pose_encoding import decode_pose
        v, _, h, w = images.shape
        gh, gw = h//16, w//16
        if h % 16 or w % 16:
            raise ValueError('DVGT-2 image height/width must be divisible by 16')
        p = gh*gw + self.core.aggregator.patch_start_idx
        past = trim_kv(state.kv, self.cfg.window-1, p)
        enabled = images.is_cuda and self.cfg.amp
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.finetune_last_blocks > 0):
            with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=enabled):
                tokens, start, cache = self.core.aggregator(images[None, None], past_key_values=past,
                    use_cache=True, past_frame_idx=state.frame_index)
            curr = tokens[-1][:, :, :, :self.core.ego_pose_window].float()
            if state.ego_tokens is None or self.cfg.window == 1:
                ego = curr
            else:
                ego = torch.cat((state.ego_tokens[:, -(self.cfg.window-1):], curr), 1)
            # Upstream head does an in-place += positional embedding. Never pass our cache directly.
            devices = [images.device.index or 0] if images.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.cfg.seed + state.frame_index)
                pose_outputs = self.core.ego_pose_head(ego.clone(), None)
            enc = self.core.ego_pose_head.post_process_pose_for_eval(pose_outputs)[:, -1]
            relative, _ = decode_pose(enc, pose_encoding_type='absT_quaR')
            relative = relative[0].clone()
            relative[:3, 3] /= self.cfg.metric_scale
            pts, conf = self.core.point_head([x.float() for x in tokens], images[None, None],
                                            patch_start_idx=start, frames_chunk_size=1)
            basis = rdf_to_flu(relative)
            relative = basis @ relative @ basis.T
            pts = pts[0, 0].float() @ basis[:3, :3].T / self.cfg.metric_scale
            features = [tokens[i][0, 0, :, start:].transpose(-1, -2).reshape(v, self.feature_dim, gh, gw).float()
                        for i in (4, 11, 17, 23)]
        nxt = BackboneState(trim_kv(cache, self.cfg.window, p), ego.detach().clone(), None, state.frame_index+1)
        return dict(features=features, points_ego=pts, point_confidence=conf[0, 0].float(),
                    previous_to_current=relative, pretrained=True), nxt


class TinyBackend(nn.Module):
    pretrained = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.feature_dim, self.patch_size = cfg, cfg.tiny_dim, cfg.patch_size
        d = cfg.tiny_dim
        self.encoder = nn.Conv2d(3, d, self.patch_size, self.patch_size)
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU()) for _ in range(4)])
        self.depth = nn.Conv2d(d, 1, 1)
        self.pose = nn.Linear(d, 3)
        nn.init.zeros_(self.pose.weight)
        nn.init.zeros_(self.pose.bias)
        self.load_report = {'pretrained': False, 'purpose': 'synthetic integration tests only'}

    def step(self, images, intrinsics, extrinsics, state):
        x = self.encoder(images)
        history = [] if state.tiny_history is None else state.tiny_history
        if history:
            x = x + torch.stack(history).mean(0)*.2
        features = []
        for layer in self.layers:
            x = x + layer(x)
            features.append(x)
        depth = (F.softplus(F.interpolate(self.depth(x), images.shape[-2:], mode='bilinear', align_corners=False)[:, 0])+1)*5
        points = unproject_depth(depth, intrinsics, extrinsics)
        relative = torch.eye(4, device=images.device, dtype=images.dtype)
        relative[:3, 3] = self.pose(x.mean((0, 2, 3)))
        keep = self.cfg.window-1
        history = (history+[x.detach().clone()])[-keep:] if keep else []
        nxt = BackboneState(tiny_history=history, frame_index=state.frame_index+1)
        return dict(features=features, points_ego=points, point_confidence=torch.ones_like(depth),
                    previous_to_current=relative, pretrained=False), nxt


def make_backend(cfg):
    return TinyBackend(cfg) if cfg.name == 'tiny' else DVGT2Backend(cfg)
