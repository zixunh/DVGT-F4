"""Foundation4D-inspired heads on DVGT-2's four intermediate feature taps.

Projection attention is a portable sampled variant, not the paper's full detector.
Occupancy uses bounded local XY pillars; decoder layers have auxiliary supervision.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import invert_pose, scale_intrinsics, unproject_depth, normalize_quaternion, matrix_to_quaternion, quaternion_multiply


class FeaturePyramid(nn.Module):
    def __init__(self, input_dim, channels):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(input_dim, channels, 1) for _ in range(4)])
        self.refine = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                                 nn.Conv2d(channels, channels, 3, padding=1)) for _ in range(4)])

    def forward(self, features):
        pyramid, prior = [], None
        # Deep coarse -> shallow fine; use distinct layer features, not four copies of the last layer.
        h, w = features[0].shape[-2:]
        for feat, proj, refine, scale in zip(reversed(features), self.projections, self.refine, (.5, 1., 2., 4.)):
            hw = (max(1, round(h*scale)), max(1, round(w*scale)))
            x = F.interpolate(proj(feat), hw, mode='bilinear', align_corners=False)
            if prior is not None:
                x = x + F.interpolate(prior, hw, mode='bilinear', align_corners=False)
            prior = x + refine(x)
            pyramid.append(prior)
        return pyramid


class ProjectedAttention(nn.Module):
    def __init__(self, dim, samples):
        super().__init__()
        self.samples = samples
        self.offsets = nn.Linear(dim, samples*3)
        self.query, self.key, self.value, self.out = [nn.Linear(dim, dim) for _ in range(4)]
        nn.init.zeros_(self.offsets.weight)
        nn.init.normal_(self.offsets.bias, std=.1)

    def forward(self, queries, refs, pyramid, k, e, hw):
        # Queries [Q,D], features [V,D,H,W]. All reference locations are metric ego FLU.
        q, d = queries.shape
        v = len(k)
        locations = refs[:, None] + self.offsets(queries).reshape(q, self.samples, 3).tanh()*4
        inv = invert_pose(e)
        camera = torch.einsum('vij,qsj->vqsi', inv[:, :3, :3], locations) + inv[:, None, None, :3, 3]
        uvz = torch.einsum('vij,vqsj->vqsi', k, camera)
        uv = uvz[..., :2] / uvz[..., 2:].clamp_min(1e-6)
        grid = 2*(uv+.5)/uv.new_tensor((hw[1], hw[0]))-1
        mask = (camera[..., 2] > .01) & (grid.abs() <= 1).all(-1)
        sampled = [F.grid_sample(f, grid, align_corners=False).permute(2, 0, 3, 1) for f in pyramid]
        sampled = torch.stack(sampled, 2).reshape(q, -1, d)
        valid = mask.permute(1, 0, 2)[:, :, None].expand(-1, -1, len(pyramid), -1).reshape(q, -1)
        score = (self.query(queries)[:, None]*self.key(sampled)).sum(-1)/d**.5
        weights = score.masked_fill(~valid, -1e4).softmax(-1)*valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        return self.out((weights[..., None]*self.value(sampled)).sum(1))


class ProjectionLayer(nn.Module):
    def __init__(self, cfg, self_attention=True):
        super().__init__()
        d = cfg.channels
        self.attn = nn.MultiheadAttention(d, cfg.attention_heads, batch_first=True) if self_attention else None
        self.cross = ProjectedAttention(d, cfg.samples)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, q, pos, refs, pyramid, k, e, hw):
        if self.attn is not None:
            v = self.norms[0](q)
            q = q + self.attn((v+pos)[None], (v+pos)[None], v[None], need_weights=False)[0][0]
        q = q + self.cross(self.norms[1](q)+pos, refs, pyramid, k, e, hw)
        return q + self.ffn(self.norms[2](q))


class DetectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reference_logits = nn.Parameter(torch.empty(cfg.detection_queries, 3).uniform_(-1.5, 1.5))
        self.position = nn.Sequential(nn.Linear(3, cfg.channels), nn.GELU(), nn.Linear(cfg.channels, cfg.channels))
        self.layers = nn.ModuleList([ProjectionLayer(cfg) for _ in range(cfg.detection_layers)])
        self.classes = nn.ModuleList([nn.Linear(cfg.channels, cfg.detection_classes) for _ in self.layers])
        self.regression = nn.ModuleList([nn.Linear(cfg.channels, 10) for _ in self.layers])
        for layer in self.classes:
            nn.init.constant_(layer.bias, -4.595)

    def forward(self, pyramid, k, e, hw):
        bounds = k.new_tensor(self.cfg.detection_bounds)
        lo, span = bounds[:3], bounds[3:]-bounds[:3]
        refs = self.reference_logits.sigmoid()
        q = k.new_zeros(self.cfg.detection_queries, self.cfg.channels)
        outputs = []
        for layer, cls, reg in zip(self.layers, self.classes, self.regression):
            q = layer(q, self.position(refs), lo+refs*span, pyramid, k, e, hw)
            raw = reg(q)
            refs = (torch.logit(refs.clamp(1e-5, 1-1e-5))+raw[:, :3]).sigmoid()
            angle = F.normalize(raw[:, 6:8]+raw.new_tensor([0., 1.]), dim=-1)
            encoded = torch.cat((lo+refs*span, raw[:, 3:6].clamp(-4, 4), angle, raw[:, 8:10]), -1)
            boxes = torch.cat((encoded[:, :3], encoded[:, 3:6].exp(), torch.atan2(angle[:, :1], angle[:, 1:2]), encoded[:, 8:10]), -1)
            outputs.append(dict(class_logits=cls(q), encoded_boxes=encoded, boxes=boxes))
        return outputs


class OccupancyHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(3, cfg.channels)
        self.position = nn.Linear(3, cfg.channels)
        # No quadratic all-voxel self attention; local spatial refinement is convolutional.
        self.layers = nn.ModuleList([ProjectionLayer(cfg, self_attention=False) for _ in range(cfg.occupancy_layers)])
        self.merge = nn.Linear(4*cfg.channels, cfg.channels)
        self.decode = nn.Sequential(nn.Conv2d(cfg.channels, cfg.channels, 3, padding=1), nn.GELU())
        nz = cfg.occupancy_shape[2]
        self.binary = nn.Conv2d(cfg.channels, nz, 1)
        self.semantic = nn.Conv2d(cfg.channels, nz*cfg.occupancy_classes, 1)
        self.bev = nn.Conv2d(cfg.channels, cfg.occupancy_classes, 1)

    def decode_grid(self, q):
        nx, ny = self.cfg.occupancy_coarse
        vx, vy, vz = self.cfg.occupancy_shape
        x = self.merge(q.reshape(nx, ny, -1)).permute(2, 0, 1)[None]
        x = self.decode(F.interpolate(x, (vx, vy), mode='bilinear', align_corners=False))
        return dict(binary_logits=self.binary(x)[0].permute(1, 2, 0),
                    semantic_logits=self.semantic(x)[0].reshape(self.cfg.occupancy_classes, vz, vx, vy).permute(2, 3, 1, 0),
                    bev_logits=self.bev(x)[0].permute(1, 2, 0))

    def forward(self, pyramid, k, e, hw):
        nx, ny = self.cfg.occupancy_coarse
        grid = torch.meshgrid(*[(torch.arange(n, device=k.device, dtype=k.dtype)+.5)/n for n in (nx, ny, 4)], indexing='ij')
        norm = torch.stack(grid, -1).reshape(-1, 3)
        bounds = k.new_tensor(self.cfg.occupancy_bounds)
        refs = bounds[:3]+norm*(bounds[3:]-bounds[:3])
        q, pos = self.embed(norm), self.position(norm)
        outputs = []
        for layer in self.layers:
            q = layer(q, pos, refs, pyramid, k, e, hw)
            outputs.append(self.decode_grid(q))
        return outputs


class GaussianHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attributes = nn.Sequential(nn.Conv2d(cfg.channels, cfg.channels, 3, padding=1), nn.GELU(), nn.Conv2d(cfg.channels, 14, 1))
        # Independent bounded residual initialized at the pretrained geometry depth.
        self.depth_residual = nn.Conv2d(cfg.channels, 1, 1)
        nn.init.zeros_(self.depth_residual.weight)
        nn.init.zeros_(self.depth_residual.bias)
        self.sky = nn.Linear(cfg.channels, 6*3*cfg.sky_resolution**2)
        self.appearance = nn.Linear(cfg.channels, 12)
        nn.init.zeros_(self.appearance.weight)
        nn.init.zeros_(self.appearance.bias)

    def forward(self, feature, images, depth, k, e, semantics):
        h, w = images.shape[-2:]
        hw = (max(1, h//self.cfg.gaussian_stride), max(1, w//self.cfg.gaussian_stride))
        feat = F.interpolate(feature, hw, mode='bilinear', align_corners=False)
        prior = F.interpolate(depth[:, None].detach(), hw, mode='bilinear', align_corners=False)[:, 0]
        gs_depth = prior.clamp_min(.05)*torch.exp(.5*self.depth_residual(feat)[:, 0].tanh())
        attr = self.attributes(feat).movedim(1, -1)
        rgb = F.interpolate(images, hw, mode='bilinear', align_corners=False).movedim(1, -1)
        local_q = normalize_quaternion(attr[..., 6:10]+attr.new_tensor([1., 0., 0., 0.]))
        ego_q = quaternion_multiply(matrix_to_quaternion(e[:, :3, :3])[:, None, None], local_q)
        normals = F.normalize(torch.einsum('vij,vhwj->vhwi', e[:, :3, :3], attr[..., 11:14]), dim=-1)
        opacity = attr[..., 10].sigmoid()
        labels = F.interpolate(semantics.detach(), hw, mode='bilinear', align_corners=False).argmax(1)
        if self.cfg.sky_class >= 0:
            opacity = opacity * (labels != self.cfg.sky_class)
        valid = torch.isfinite(prior) & (prior > .05)
        opacity = opacity*valid
        pooled = feature.mean((-1, -2))
        appearance = .1*self.appearance(pooled).tanh()
        return dict(means=unproject_depth(gs_depth, scale_intrinsics(k, (h, w), hw), e),
                    depth=gs_depth, colors=(rgb+.1*attr[..., :3].tanh()).clamp(0, 1),
                    scales=F.softplus(attr[..., 3:6])*.1+1e-4, quaternions=ego_q,
                    opacities=opacity, normals=normals, semantic_labels=labels,
                    sky_cubemap=self.sky(pooled.mean(0)).reshape(6, 3, self.cfg.sky_resolution, self.cfg.sky_resolution).sigmoid(),
                    affine_matrix=torch.eye(3, device=images.device)[None]+appearance[:, :9].reshape(-1, 3, 3),
                    affine_bias=appearance[:, 9:12])


class TaskHeads(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.pyramid = FeaturePyramid(input_dim, cfg.channels)
        self.semantic = nn.Conv2d(cfg.channels, cfg.semantic_classes, 1)
        self.detection = DetectionHead(cfg)
        self.occupancy = OccupancyHead(cfg)
        self.gaussian = GaussianHead(cfg)

    def forward(self, features, images, depth, k, e):
        pyramid = self.pyramid(features)
        semantics = F.interpolate(self.semantic(pyramid[-1]), images.shape[-2:], mode='bilinear', align_corners=False)
        return dict(semantic_logits=semantics,
                    detections=self.detection(pyramid, k, e, images.shape[-2:]),
                    occupancy=self.occupancy(pyramid, k, e, images.shape[-2:]),
                    gaussians=self.gaussian(pyramid[-1], images, depth, k, e, semantics))
