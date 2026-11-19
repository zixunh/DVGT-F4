import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch

from .checkpoint import load_adapter, save_adapter
from .config import load_config
from .data import iter_frames, make_synthetic, to_device
from .losses import multitask_loss
from .model import StreamingFoundation4D
from .geometry import scale_intrinsics
from .render import render_scene


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def make_model(args):
    if args.checkpoint:
        model,saved=load_adapter(args.checkpoint,args.device)
    else:
        model=StreamingFoundation4D(load_config(args.config)).to(args.device)
        saved={}
    return model,saved


def check_source(frame,model,args):
    if model.config.backend.name=='tiny' and not frame.get('synthetic',False):
        raise ValueError('Tiny backend is restricted to declared synthetic integration fixtures')
    if frame.get('synthetic',False) and not getattr(args,'allow_synthetic',False):
        raise ValueError('Synthetic data requires --allow-synthetic')
    if getattr(model,'synthetic_checkpoint',False) and not frame.get('synthetic',False):
        raise ValueError('A synthetic checkpoint cannot be used as a real-data baseline')


def train(args):
    torch.manual_seed(args.seed)
    model,saved=make_model(args);model.train();model.set_stage(args.stage)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    perceptual=None
    if args.lpips:
        import lpips
        perceptual=lpips.LPIPS(net='alex').to(args.device).eval().requires_grad_(False)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    steps=0;first_loss=None;last_loss=None;synthetic=False
    started=time.monotonic()
    with (root/'train.jsonl').open('w') as log:
        while steps<args.steps:
            state=None;scene=None;count=0
            for frame in iter_frames(args.manifest):
                check_source(frame,model,args)
                if scene!=frame['scene_id']: state=None;scene=frame['scene_id']
                synthetic=synthetic or bool(frame.get('synthetic',False))
                frame=to_device(frame,args.device)
                optimizer.zero_grad(set_to_none=True)
                pred,state=model(frame,state)
                loss,detail=multitask_loss(pred,frame,model.config,args.stage,perceptual)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {steps}')
                if not loss.requires_grad: raise RuntimeError('Provided supervision has no trainable path in this stage')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();steps+=1;count+=1
                values={k:float(v.detach()) for k,v in detail.items()}
                record=dict(step=steps,**values);log.write(json.dumps(record)+'\n');log.flush()
                if first_loss is None:first_loss=values['total']
                last_loss=values['total']
                if steps==1 or steps%10==0 or steps==args.steps: print(json.dumps(record),flush=True)
                if steps>=args.steps:break
            if count==0:raise ValueError('Training manifest contains no frames')
    save_adapter(root/'adapter.pt',model,args.stage,steps,synthetic)
    summary=dict(stage=args.stage,steps=steps,first_loss=first_loss,last_loss=last_loss,
                 elapsed_seconds=time.monotonic()-started,synthetic=synthetic,
                 trainable_parameters=sum(p.numel() for p in params),
                 pretrained_geometry=model.backend.pretrained,lpips=args.lpips)
    json_write(root/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def prediction_arrays(pred):
    geo=pred['geometry'];det=pred['detections'][-1];occ=pred['occupancy'][-1]
    values={f'geometry/{k}':v for k,v in geo.items()}
    values.update({'semantic_labels':pred['semantic_logits'].argmax(1),'boxes_ego':det['boxes'],
                   'box_probabilities':det['class_logits'].sigmoid(),
                   'occupancy_probability':occ['binary_logits'].sigmoid(),
                   'occupancy_semantic':occ['semantic_logits'].argmax(-1)})
    values.update({f'tracks/{k}':v for k,v in pred['tracks'].items()})
    return {k:v.detach().cpu().numpy() for k,v in values.items()}


def infer(args):
    torch.manual_seed(args.seed)
    model,saved=make_model(args);model.eval()
    if not args.checkpoint and not args.allow_untrained_heads:
        raise ValueError('New task heads need an adapter checkpoint; use --allow-untrained-heads only to test interfaces')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    state=None;scene=None;count=0;peak_state=0;last_stats={};depth_abs=0.;depth_count=0
    started=time.monotonic()
    with torch.inference_mode(),(root/'stream.jsonl').open('w') as log:
        frames=iter_frames(args.manifest)
        if args.max_frames:frames=itertools.islice(frames,args.max_frames)
        for frame in frames:
            check_source(frame,model,args)
            if scene!=frame['scene_id']:state=None;scene=frame['scene_id']
            frame=to_device(frame,args.device)
            pred,state=model.step(frame,state)
            arrays=prediction_arrays(pred)
            if args.render:
                v=0;hw=tuple(args.render_size)
                gs=pred['gaussians']
                image=render_scene(state.gaussian_map.scene(pred['tracks']),
                    scale_intrinsics(frame['intrinsics'][v],frame['images'].shape[-2:],hw),
                    pred['geometry']['camera_to_world'][v],hw,backend=model.config.renderer,
                    sky_cubemap=gs['sky_cubemap'],sky_to_world=pred['geometry']['world_from_ego'],
                    affine_matrix=gs['affine_matrix'][v],affine_bias=gs['affine_bias'][v])
                arrays.update({f'render/{k}':v.cpu().numpy() for k,v in image.items()})
            np.savez_compressed(root/f'frame_{count:06d}.npz',**arrays)
            stats=dict(scene_id=scene,**pred['metadata'])
            log.write(json.dumps(stats)+'\n');log.flush()
            peak_state=max(peak_state,stats['backbone_bytes']);last_stats=stats
            if 'depth_z' in frame:
                gt=frame['depth_z'];valid=torch.isfinite(gt)&(gt>0)
                depth_abs+=float((pred['geometry']['depth_z'][valid]-gt[valid]).abs().sum())
                depth_count+=int(valid.sum())
            count+=1
    if not count:raise ValueError('Inference manifest contains no frames')
    summary=dict(frames=count,elapsed_seconds=time.monotonic()-started,peak_backbone_state_bytes=peak_state,
                 last_state=last_stats,pretrained_geometry=model.backend.pretrained,
                 trained_stages=saved.get('trained_stages',[]),synthetic_checkpoint=saved.get('synthetic',False),
                 depth_mae_m=None if not depth_count else depth_abs/depth_count,
                 peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None)
    json_write(root/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description='DVGT-F4 online multi-task baseline')
    sub=parser.add_subparsers(dest='command',required=True)
    synth=sub.add_parser('synthetic');synth.add_argument('--config',default='configs/stream4d/tiny.yaml')
    synth.add_argument('--output',required=True);synth.add_argument('--frames',type=int,default=16)
    synth.add_argument('--size',nargs=2,type=int,default=[32,48]);synth.add_argument('--views',type=int,default=2)
    for name in ('train','infer'):
        p=sub.add_parser(name);p.add_argument('--config',default='configs/stream4d/dvgt2.yaml')
        p.add_argument('--checkpoint');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
        p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--seed',type=int,default=17);p.add_argument('--allow-synthetic',action='store_true')
        if name=='train':
            p.add_argument('--stage',choices=['perception','gaussian','joint'],default='perception')
            p.add_argument('--steps',type=int,default=100);p.add_argument('--lr',type=float,default=1e-4)
            p.add_argument('--lpips',action='store_true')
        else:
            p.add_argument('--allow-untrained-heads',action='store_true');p.add_argument('--max-frames',type=int)
            p.add_argument('--render',action='store_true');p.add_argument('--render-size',type=int,nargs=2,default=[32,48])
    args=parser.parse_args();torch.set_num_threads(min(torch.get_num_threads(),8))
    if args.command=='synthetic':
        print(make_synthetic(args.output,load_config(args.config),args.frames,args.views,tuple(args.size)))
    elif args.command=='train':
        if args.steps<1:parser.error('--steps must be positive')
        train(args)
    else:infer(args)


if __name__=='__main__':main()
