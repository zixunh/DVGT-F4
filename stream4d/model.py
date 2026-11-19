import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from .backends import make_backend
from .config import Config
from .geometry import invert_pose, ego_points_to_camera, transform_boxes
from .heads import TaskHeads
from .state import BackboneState, StreamState
from .tracking import OnlineTracker
from .gaussian_map import GaussianMap


class StreamingFoundation4D(nn.Module):
    """One independent stream per state; camera batches are synchronized multi-view frames.

    step() returns the current frame immediately. Pass state=None to reset a clip.
    Gradients can flow through current features/heads; stored histories are detached.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = (config or Config()).validate()
        self.backend = make_backend(self.config.backend)
        self.heads = TaskHeads(self.backend.feature_dim, self.config.heads)

    def new_state(self):
        return StreamState(BackboneState(), OnlineTracker(self.config.stream),
                           GaussianMap(self.config.stream, self.config.heads.dynamic_classes))

    def validate_frame(self, frame, state):
        images, k, e = frame['images'], frame['intrinsics'], frame['camera_to_ego']
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError('images must be floating [V,3,H,W] in [0,1]')
        v, _, h, w = images.shape
        if min(v,h,w) < 1 or k.shape != (v,3,3) or e.shape != (v,4,4):
            raise ValueError('Expected intrinsics [V,3,3], camera_to_ego [V,4,4]')
        if len({images.device,k.device,e.device}) != 1:
            raise ValueError('Images and calibration must use the same device')
        if not all(torch.isfinite(x).all() for x in (images,k,e)) or images.min()<0 or images.max()>1:
            raise ValueError('Input/calibration contains invalid values')
        if h % self.backend.patch_size or w % self.backend.patch_size:
            raise ValueError(f'Image dimensions must be divisible by {self.backend.patch_size}')
        if not torch.allclose(k[:,2], k.new_tensor([0.,0.,1.]).expand(v,-1), atol=1e-5) or (k[:,[0,1],[0,1]]<=0).any():
            raise ValueError('Only calibrated rectified pinhole images are supported')
        if not torch.allclose(e[:,3], e.new_tensor([0.,0.,0.,1.]).expand(v,-1), atol=1e-5):
            raise ValueError('Invalid homogeneous camera_to_ego matrices')
        r=e[:,:3,:3]
        if not torch.allclose(r.transpose(-1,-2)@r, torch.eye(3,device=e.device).expand(v,-1,-1),atol=1e-3) or (torch.linalg.det(r)<.99).any():
            raise ValueError('camera_to_ego rotations must belong to SO(3)')
        timestamp = float(frame['timestamp'])
        if not torch.isfinite(torch.tensor(timestamp, dtype=torch.float64)):
            raise ValueError('Timestamp must be finite relative seconds')
        if state.timestamp is not None:
            if timestamp <= state.timestamp:
                raise ValueError('Timestamps must strictly increase; reset state at clip boundaries')
            if timestamp-state.timestamp > self.config.stream.max_gap_seconds:
                raise ValueError('Frame gap exceeds configured limit; reset the stream or increase max_gap_seconds')
        times = torch.as_tensor(frame.get('camera_timestamps', [timestamp]*v), dtype=torch.float64)
        if times.shape != (v,) or not torch.isfinite(times).all() or (times-timestamp).abs().max() > self.config.stream.max_camera_skew_seconds:
            raise ValueError('Camera skew exceeds synchronization tolerance; align cameras before step()')
        ids = tuple(frame.get('camera_ids', [str(i) for i in range(v)]))
        if len(ids)!=v or len(set(ids))!=v:
            raise ValueError('camera_ids must contain one unique ID per view')
        digest = hashlib.sha256(k.detach().float().cpu().numpy().tobytes()+e.detach().float().cpu().numpy().tobytes()).hexdigest()
        signature = (frame.get('scene_id','default'), ids, tuple(images.shape), str(images.device), digest)
        if state.signature is not None and signature != state.signature:
            raise ValueError('Scene/camera order/calibration/resolution changed; reset stream state')
        return timestamp, signature

    def step(self, frame, state=None, *, update_map=True):
        state = self.new_state() if state is None else state
        timestamp, signature = self.validate_frame(frame, state)
        images, k, e = frame['images'], frame['intrinsics'].float(), frame['camera_to_ego'].float()
        raw, next_backbone = self.backend.step(images, k, e, state.backbone)
        points = raw['points_ego'].float()
        camera = ego_points_to_camera(points, e)
        valid = torch.isfinite(camera).all(-1) & (camera[...,2]>.01)
        depth = torch.where(valid, camera[...,2], 0.)
        previous_to_current = raw['previous_to_current'].float()
        if state.world_from_ego is None:
            world = torch.eye(4,device=images.device)
        else:
            # Delta maps previous ego -> current ego, hence the inverse here.
            world = state.world_from_ego @ invert_pose(previous_to_current)
        geometry = dict(points_ego=points, points_world=points @ world[:3,:3].T+world[:3,3],
                        depth_z=depth, camera_range=torch.where(valid,camera.norm(dim=-1),0.),
                        point_confidence=raw['point_confidence'], valid=valid,
                        previous_to_current=previous_to_current, world_from_ego=world,
                        camera_to_world=world[None]@e)
        result = self.heads(raw['features'], images.float(), depth, k, e)
        result['geometry'] = geometry
        detection = result['detections'][-1]
        scores, labels = detection['class_logits'].sigmoid().max(-1)
        accepted = (scores>=self.config.stream.score_threshold) & torch.isfinite(detection['boxes']).all(-1)
        world_boxes = transform_boxes(detection['boxes'][accepted].detach(), world.detach())
        # Keep discrete tracking and the long-lived map outside the autograd graph.
        state.tracker.update(world_boxes, labels[accepted], scores[accepted], timestamp)
        tracks = state.tracker.tensors(images.device)
        result['tracks'] = tracks
        gs = result['gaussians']
        confidence = F.interpolate(raw['point_confidence'][:,None], gs['depth'].shape[-2:], mode='nearest')[:,0]
        gs['opacities'] = gs['opacities']*(confidence>=self.config.stream.min_point_confidence)
        if update_map:
            state.gaussian_map.update(gs, world.detach(), tracks)
        state.backbone = next_backbone.detached()
        state.world_from_ego = world.detach().clone()
        state.timestamp, state.signature = timestamp, signature
        result['metadata'] = dict(timestamp=timestamp, pretrained_geometry=self.backend.pretrained,
                                  new_heads_trained=False, **state.statistics())
        return result, state

    def forward(self, frame, state=None):
        return self.step(frame, state, update_map=False)

    def stream(self, frames, state=None):
        for frame in frames:
            result, state = self.step(frame,state)
            yield result, state

    def set_stage(self, stage):
        if stage not in ('perception','gaussian','joint'):
            raise ValueError('stage must be perception, gaussian, or joint')
        self.heads.requires_grad_(stage!='gaussian')
        self.heads.gaussian.requires_grad_(stage!='perception')
        if stage=='gaussian':
            self.backend.requires_grad_(False)
