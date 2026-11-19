import unittest
import torch
from stream4d.geometry import (invert_pose, rdf_to_flu, camera_rays, scale_intrinsics,
                               quaternion_to_matrix, matrix_to_quaternion, assign_points_to_boxes)
from stream4d.backends import trim_kv


class GeometryTests(unittest.TestCase):
    def test_pose_direction_and_basis(self):
        delta=torch.eye(4); delta[2,3]=-2
        basis=rdf_to_flu(delta)
        world=invert_pose(basis@delta@basis.T)
        torch.testing.assert_close(world[:3,3],torch.tensor([2.,0.,0.]))

    def test_quaternion_half_turn(self):
        q=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        r=quaternion_to_matrix(q)
        torch.testing.assert_close(quaternion_to_matrix(matrix_to_quaternion(r)),r)

    def test_pixel_center_resize(self):
        k=torch.tensor([[[50.,0.,31.5],[0.,50.,15.5],[0.,0.,1.]]])
        scaled=scale_intrinsics(k,(32,64),(16,32))
        torch.testing.assert_close(scaled[0,:2,2],torch.tensor([15.5,7.5]))
        self.assertEqual(camera_rays(scaled,(16,32)).shape,(1,16,32,3))

    def test_rotated_box_assignment(self):
        boxes=torch.tensor([[0.,0.,0.,4.,1.,2.,torch.pi/2,0.,0.]])
        pts=torch.tensor([[0.,1.5,0.],[1.5,0.,0.]])
        self.assertEqual(assign_points_to_boxes(pts,boxes).tolist(),[0,-1])

    def test_cache_eviction_and_window_one(self):
        x=torch.randn(2,3,30,4)
        cropped=trim_kv([(x,x)],2,5)[0][0]
        self.assertEqual(cropped.shape[-2],10)
        self.assertLess(cropped.untyped_storage().nbytes(),x.untyped_storage().nbytes())
        self.assertIsNone(trim_kv([(x,x)],0,5))


if __name__=='__main__': unittest.main()
