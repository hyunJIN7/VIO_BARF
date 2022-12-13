import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle
from . import base
import camera
from util import log,debug
from scipy.spatial.transform import Rotation
import cv2
from PIL import Image
import skvideo.io

class Dataset(base.Dataset):
    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 192,256
        super().__init__(opt,split)
        self.root = opt.data.root or "data/strayscanner"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.split = split
        # load/parse metadata
        intrin_file = os.path.join(self.path, 'camera_matrix.csv')
        assert os.path.isfile(intrin_file), "camera info:{} not found".format(intrin_file)
        intrinsics = np.loadtxt(intrin_file, delimiter=',')
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()
        intrinsics[0, :] = intrinsics[0, :] * float(self.raw_W) / 1920.0
        intrinsics[1, :] = intrinsics[1, :] * float(self.raw_H) / 1440.0
        self.intr = intrinsics

        pose_path = "{}/odometry_{}.csv".format(self.path,split)
        assert os.path.isfile(pose_path), "pose info:{} not found".format(pose_path)
        odometry = np.loadtxt(pose_path, delimiter=',')#, skiprows=1
        self.frames = odometry
        poses = []
        for line in odometry: # timestamp, frame(float ex 1.0), x, y, z, qx, qy, qz, qw
            position = line[2:5]
            quaternion = line[5:]
            T_WC = np.eye(4)
            T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            T_WC[:3, 3] = position
            poses.append(T_WC)
        poses = torch.from_numpy(np.array(poses)).float()
        self.list = poses
        self.gt_pose = poses
        self.opti_pose = poses

        # if subset and split != 'test': self.list = self.list[:subset] #train,val
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
        # for s in self:
        #     print()

    def get_all_depth(self,opt):
        depth = torch.stack([torch.tensor(f ,dtype=torch.float32) for f in self.depth])
        confidence = torch.stack([torch.tensor(f, dtype=torch.float32) for f in self.confidence])
        return depth,confidence

    #get_all_gt_camera_poses
    def get_all_gt_depth(self,opt): # optitrack pose load
        depth = torch.stack([torch.tensor(f, dtype=torch.float32) for f in self.depth])
        return depth

    def get_all_camera_poses(self,opt):
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.list] # """list : campose 의미"""
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    #get_all_gt_camera_poses
    def get_all_gt_camera_poses(self,opt): # optitrack pose load
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.gt_pose]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    def get_all_optitrack_camera_poses(self,opt): # optitrack pose load
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.opti_pose]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None

        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)

        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx) #(3,4)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)

        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}.png".format(str(int(self.frames[idx][1])).zfill(5))
        image_fname = "{}/rgb_{}/{}".format(self.path,self.split,image_fname)
        image = PIL.Image.fromarray(imageio.imread(image_fname))
        return image


    def get_camera(self,opt,idx):
        intrinsics = self.intr
        pose_raw = torch.tensor(self.list[idx],dtype=torch.float32)
        pose = self.parse_raw_camera(opt,pose_raw) #pose_raw (3,4)
        return intrinsics,pose

    # [right, forward, up]
    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,1,1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])  # [right, down, forward] , pose_raw[:3]=pose_flip=(3,4),(3,4)
        pose = camera.pose.invert(pose)
        return pose
