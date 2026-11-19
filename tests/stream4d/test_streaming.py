import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch

from stream4d.config import load_config, ROOT
from stream4d.model import StreamingFoundation4D
from stream4d.geometry import rdf_to_flu, box_poses
from stream4d.gaussian_map import GaussianMap, move_actors
from stream4d.losses import occupancy_loss, multitask_loss
from stream4d.data import iter_frames, make_synthetic
from stream4d.tracking import OnlineTracker
from stream4d.checkpoint import save_adapter, load_adapter


def config():return load_config(ROOT/'configs/stream4d/tiny.yaml')


def frame(t=0.,seed=11):
    g=torch.Generator().manual_seed(seed)
    images=torch.rand(2,3,32,48,generator=g)
    k=torch.tensor([[40.,0.,23.5],[0.,40.,15.5],[0.,0.,1.]]).repeat(2,1,1)
    e=rdf_to_flu(k).repeat(2,1,1)
    return dict(images=images,intrinsics=k,camera_to_ego=e,timestamp=t,scene_id='test',camera_ids=['left','right'])


def gaussians(means):
    n=len(means)
    return dict(means=means,scales=torch.ones(n,3)*.1,quaternions=torch.tensor([1.,0.,0.,0.]).repeat(n,1),
                colors=torch.ones(n,3)*.5,opacities=torch.ones(n)*.8,normals=torch.tensor([0.,0.,1.]).repeat(n,1))


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.cfg=config();self.model=StreamingFoundation4D(self.cfg).eval()

    def test_causality_and_chunk_boundaries(self):
        with torch.no_grad():
            state=None;prefix=[]
            for i in range(5):
                out,state=self.model.step(frame(i*.5,seed=i),state)
                prefix.append(out['semantic_logits'].clone())
            state=None
            # A caller may stop and resume with the same state; future images differ.
            for i in range(7):
                out,state=self.model.step(frame(i*.5,seed=i if i<5 else 1000+i),state)
                if i<5:torch.testing.assert_close(out['semantic_logits'],prefix[i],rtol=0,atol=0)
            # Detached historical state must not retain an ever-growing autograd graph.
            for x in state.backbone.tiny_history:self.assertIsNone(x.grad_fn)

    def test_long_video_bounded_state(self):
        sizes=[];state=None
        with torch.no_grad():
            for i in range(80):
                out,state=self.model.step(frame(i*.5,seed=i),state)
                stats=state.statistics();sizes.append(stats['backbone_bytes'])
                self.assertLessEqual(stats['static_gaussians'],self.cfg.stream.static_gaussians)
                self.assertLessEqual(stats['actor_gaussians'],self.cfg.stream.actor_gaussians*self.cfg.stream.track_max_count)
                self.assertLessEqual(stats['tracks'],self.cfg.stream.track_max_count)
        self.assertEqual(len(set(sizes[3:])),1)

    def test_stream_invalidations(self):
        with torch.no_grad():out,state=self.model.step(frame())
        for changed in ('time','camera','scene','calibration','skew'):
            f=frame(.5)
            if changed=='time':f['timestamp']=0
            if changed=='camera':f['camera_ids'].reverse()
            if changed=='scene':f['scene_id']='new'
            if changed=='calibration':f['intrinsics'][0,0,0]+=1
            if changed=='skew':f['camera_timestamps']=[.5,.7]
            with self.assertRaises(ValueError):self.model.step(f,state)
        self.assertEqual(state.backbone.frame_index,1)

    def test_reset_first_pose_and_window_one(self):
        cfg=config();cfg.backend.window=1;m=StreamingFoundation4D(cfg).eval()
        with torch.no_grad():
            p,s=m.step(frame());_,s=m.step(frame(.5),s)
        self.assertEqual(s.backbone.tiny_history,[])
        torch.testing.assert_close(p['geometry']['world_from_ego'],torch.eye(4))

    def test_tracker_velocity_class_gate_and_expiry(self):
        tr=OnlineTracker(self.cfg.stream)
        box=torch.tensor([[5.,0.,1.,2.,1.,2.,0.,1.,0.]])
        labels=torch.tensor([0]);scores=torch.tensor([.9])
        ids=tr.update(box,labels,scores,0.)
        moved=box.clone();moved[:,0]+=.5
        ids2=tr.update(moved,labels,scores,.5)
        self.assertEqual(ids.tolist(),ids2.tolist())
        ids3=tr.update(moved,torch.tensor([1]),scores,1.)
        self.assertNotEqual(ids.tolist(),ids3.tolist())
        tr.update(box[:0],labels[:0],scores[:0],4.)
        self.assertFalse(tr.tracks)

    def test_actor_local_map_moves_with_track_and_expires(self):
        gm=GaussianMap(self.cfg.stream)
        gs=gaussians(torch.tensor([[5.,0.,1.],[10.,0.,1.]]));gs['semantic_labels']=torch.tensor([2,1])
        box=torch.tensor([[5.,0.,1.,2.,2.,2.,0.,0.,0.]])
        tracks=dict(ids=torch.tensor([1]),boxes=box)
        gm.update(gs,torch.eye(4),tracks)
        moved=box.clone();moved[:,0]+=3
        scene=gm.scene(dict(ids=tracks['ids'],boxes=moved))
        self.assertTrue(torch.any(torch.isclose(scene['means'][:,0],torch.tensor(8.))))
        self.assertTrue(torch.any(torch.isclose(scene['means'][:,0],torch.tensor(10.))))
        empty=dict(ids=torch.empty(0,dtype=torch.long),boxes=box[:0])
        gm.update(gs,torch.eye(4),empty)
        self.assertEqual(len(gm.actors),0)

    def test_dynamic_render_transform_has_gradients(self):
        means=torch.tensor([[5.,0.,1.],[10.,0.,1.]],requires_grad=True)
        data=gaussians(means)
        source=torch.tensor([[5.,0.,1.,2.,2.,2.,0.,0.,0.]])
        target=source.clone();target[:,0]+=2
        out=move_actors(data,source,target,torch.tensor([3]),torch.tensor([3]))
        torch.testing.assert_close(out['means'][:,0],torch.tensor([7.,10.]))
        out['means'].sum().backward();self.assertTrue(torch.isfinite(means.grad).all())

    def test_unknown_occupancy_is_not_free(self):
        binary=torch.zeros(2,2,2,requires_grad=True)
        sem=torch.zeros(2,2,2,3,requires_grad=True)
        bev=torch.zeros(2,2,3,requires_grad=True)
        loss=occupancy_loss([dict(binary_logits=binary,semantic_logits=sem,bev_logits=bev)],torch.full((2,2,2),-1))
        self.assertEqual(float(loss),0.)
        loss.backward();self.assertEqual(float(binary.grad.abs().sum()),0.)

    def test_microsecond_precision_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=make_synthetic(directory,self.cfg,frames=2)
            records=[json.loads(s) for s in Path(manifest).read_text().splitlines()]
            records[1]['timestamp_us']=records[0]['timestamp_us']+123456
            Path(manifest).write_text('\n'.join(json.dumps(x) for x in records)+'\n')
            frames=list(iter_frames(manifest))
            self.assertAlmostEqual(frames[1]['timestamp'],.123456,places=12)
            path=Path(directory)/'adapter.pt'
            save_adapter(path,self.model,'perception',1,True)
            loaded,_=load_adapter(path);loaded.eval()
            with torch.no_grad():
                a,_=self.model(frame());b,_=loaded(frame())
            torch.testing.assert_close(a['semantic_logits'],b['semantic_logits'])

    def test_train_all_heads_and_heldout_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            f=next(iter_frames(make_synthetic(directory,self.cfg,frames=1)))
            self.model.train()
            out,_=self.model(f)
            loss,_=multitask_loss(out,f,self.cfg,'joint')
            self.assertTrue(torch.isfinite(loss));loss.backward()
            for name in ('semantic','detection','occupancy','gaussian'):
                grads=[p.grad for p in getattr(self.model.heads,name).parameters() if p.grad is not None]
                self.assertTrue(grads,name)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)


if __name__=='__main__':unittest.main()
