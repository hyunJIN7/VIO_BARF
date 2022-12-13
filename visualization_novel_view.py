import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import lpips

import util_vis
import camera

"""
nerf.py의 novel_pose plot 위한 코드 
"""
# python visualization_novel_view.py --expname cube
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./novel_poses',
                        help='input data directory')
    parser.add_argument("--expname", type=str, default='tri',
                        help='experiment name')
    return parser


def generate_videos_pose(pose,pose_ref):# novel pose, raw pose
    fig = plt.figure(figsize=(10,10))
    cam_path = "novel_poses"
    os.makedirs(cam_path,exist_ok=True)
    util_vis.plot_save_novel_poses(fig,pose,pose_ref=pose_ref,path=cam_path,ep=args.expname)
    plt.close()


def novel_view(args):
    rectangle_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0]]
       ).float()
    tri_prism = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3],
        [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 3],
        [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 3],
        ]
       ).float()
    cube_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3],
         [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 3],
         [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 3],
         [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 3]]
       ).float()
    small_cube_pose = torch.tensor(
        [[1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0],
         [1, 0, 0, 0.7, 0, 1, 0, 0,   0, 0, 1, 0],
         [1, 0, 0, 0.7, 0, 1, 0, 0.7, 0, 0, 1, 0],
         [1, 0, 0, 0,   0, 1, 0, 0.7, 0, 0, 1, 0],
         [1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0.8],
         [1, 0, 0, 0.4, 0, 1, 0, 0,   0, 0, 1, 0.8],
         [1, 0, 0, 0.4, 0, 1, 0, 0.7, 0, 0, 1, 0.8],
         [1, 0, 0, 0,   0, 1, 0, 0.7, 0, 0, 1, 0.8]]
       ).float()

    random_cube_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 1.1, 0, 1, 0],
         [1, 1, 0, 3, 0, 1, 0, 1, 0, 0, 1, -1],
         [1, 0, 1, 3, 0, 1, 0, -1, 0, 0, 1, 1],
         [1, 1, 1.3, 0, 0, 2, 3.2, 2.4, 0, 0, 2, 0],
         [1, 0, 2, 2.1, 0, 1, 0, -1, 0, 0, 1, 2],
         [1, 3, 0, 1, 0, 2, 0, 1.2, 0, 0, 1, -2],
         [1, 0, 2, 3, 0, 1, 0, 3, 2.1, 0, 1, -1],
         [2.4, 1.4, 2.1, 0, 2.5, 3, 0, 3, 0, 0, 1, 0]]
       ).float()

    # python visualization_novel_view.py --expname tri_prism
    main_pose = tri_prism
    poses = [torch.reshape(i,(3,4)) for i in main_pose] # list
    poses = torch.stack(poses)  # torch.tensor
    print(poses.shape)

    scale = 1

    """
        rotate novel views around the "center" camera of all poses
        모든 포즈의 센터 중심으로 새로운 뷰 회전시킴 
        tri_prism의 경우
        torch.mean :  return mean value of all elements in the input
            (6,3,4) --> (1,3,4)  각 자리 평균냄 
            tensor([[[1.0000, 0.0000, 0.0000, 2.0000],
                    [0.0000, 1.0000, 0.0000, 1.0000],
                    [0.0000, 0.0000, 1.0000, 1.5000]]])
        minus : (6,3,4) 각 pose에서 mean pose 다 뺌 
            tensor([[[ 0.0000,  0.0000,  0.0000, -2.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.5000]],
        
                [[ 0.0000,  0.0000,  0.0000,  1.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.5000]],
        
                [[ 0.0000,  0.0000,  0.0000,  1.0000],
                 [ 0.0000,  0.0000,  0.0000,  2.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.5000]],
        
                [[ 0.0000,  0.0000,  0.0000, -2.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.0000],
                 [ 0.0000,  0.0000,  0.0000,  1.5000]],
        
                [[ 0.0000,  0.0000,  0.0000,  1.0000],
                 [ 0.0000,  0.0000,  0.0000, -1.0000],
                 [ 0.0000,  0.0000,  0.0000,  1.5000]],
        
                [[ 0.0000,  0.0000,  0.0000,  1.0000],
                 [ 0.0000,  0.0000,  0.0000,  2.0000],
                 [ 0.0000,  0.0000,  0.0000,  1.5000]]])
        before_norm : (6,3) , 각 포즈별  translation만 가져옴 
            tensor([[-2.0000, -1.0000, -1.5000],
                    [ 1.0000, -1.0000, -1.5000],
                    [ 1.0000,  2.0000, -1.5000],
                    [-2.0000, -1.0000,  1.5000],
                    [ 1.0000, -1.0000,  1.5000],
                    [ 1.0000,  2.0000,  1.5000]])
        norm : return the marix norm tensor([2.6926, 2.0616, 2.6926, 2.6926, 2.0616, 2.6926])
        idx_center : (1) min값 하나 가져와서 
        pose_novel:
            center pose 위치 중심으로 N개의 각으로 나눈 후 x,y 축 기준 오일러 각도에서 회전 행렬 생성 후러, x,y 회전 행렬 합침 
            z축 +움직임 -> x,y 회전행렬 적용 -> z 축 -움직임 적용(+와 크기 다름 )        
    """
    # mean_pose_test = poses.mean(dim=0, keepdim=True)  #x
    # minus = poses - poses.mean(dim=0, keepdim=True)
    # before_norm = (poses - poses.mean(dim=0, keepdim=True))[..., 3] #x
    # norm = (poses - poses.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1)
    idx_center = (poses - poses.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
    # center_pose = poses[idx_center]
    pose_novel = camera.get_novel_view_poses(opt=None,pose_anchor= poses[idx_center], N=30, scale=scale)

    generate_videos_pose(pose_novel,poses) # novel, pose

# python visualization_novel_view.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    novel_view(args)
