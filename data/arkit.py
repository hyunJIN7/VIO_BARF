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

class Dataset(base.Dataset):
    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 480,640
        super().__init__(opt,split)
        self.root = opt.data.root or "data/arkit"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        # load/parse metadata
        pose_fname = "{}/transforms_{}.txt".format(self.path,split)
        pose_file = os.path.join('./', pose_fname)
        assert os.path.isfile(pose_file), "pose info:{} not found".format(pose_file)
        with open(pose_file, "r") as f:  # frame.txt 읽어서
            cam_frame_lines = f.readlines()
        cam_pose = []  #r1x y z tx r2x y z ty r3x y z tz
        self.frames = []  #timestamp imagenum r1x y z tx r2x y z ty r3x y z tz
        for line in cam_frame_lines:
            line_data_list = line.split(' ')
            if len(line_data_list) == 0:
                continue
            self.frames.append(line_data_list)
            pose_raw = np.reshape(line_data_list[2:] ,(3,4))
            cam_pose.append(pose_raw)
        cam_pose =  np.array(cam_pose,dtype=float)
        self.list = cam_pose
        #self.focal = 0.5*self.raw_W/np.tan(0.5*self.meta["camera_angle_x"])

        self.gt_pose = cam_pose

        self.opti_pose = cam_pose
        # for GT data(optitrack)
        gt_pose_fname = "{}/opti_transforms_{}.txt".format(self.path,split)
        gt_pose_file = os.path.join('./', gt_pose_fname)
        if os.path.isfile(gt_pose_file): # gt file exist
            print("##########opti load ########")
            with open(gt_pose_file, "r") as f:  # frame.txt 읽어서
                cam_frame_lines = f.readlines()
            cam_gt_pose = []  # time r1x y z tx r2x y z ty r3x y z tz
            for line in cam_frame_lines:
                line_data_list = line.split(' ')
                if len(line_data_list) == 0:
                    continue
                pose_raw = np.reshape(line_data_list[1:], (3, 4))
                cam_gt_pose.append(pose_raw)
            cam_gt_pose = np.array(cam_pose, dtype=float)
            self.opti_pose = cam_gt_pose
        else: self.opti_pose = cam_pose


        if subset: self.list = self.list[:subset] #train,val
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.list] # """list : campose 의미"""
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all
    #get_all_gt_camera_poses
    def get_all_gt_camera_poses(self,opt): # optitrack pose load
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.gt_pose] # """list : campose 의미"""
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    def get_all_optitrack_camera_poses(self,opt): # optitrack pose load
        pose_raw_all = [torch.tensor(f ,dtype=torch.float32) for f in self.gt_pose] # """list : campose 의미"""
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
        image_fname = "{}/{}.jpg".format(self.path,self.frames[idx][1])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image


    def get_camera(self,opt,idx):
        #Load camera intrinsics  # frane.txt -> camera intrinsics
        intrin_file = os.path.join(os.path.abspath('./'), self.path,'Frames.txt')
        assert os.path.isfile(intrin_file), "camera info:{} not found".format(intrin_file)
        with open(intrin_file, "r") as f:  # frame.txt
            cam_intrinsic_lines = f.readlines()
        cam_intrinsics = []
        line_data_list = cam_intrinsic_lines[idx].split(',')
        cam_intrinsics.append([float(i) for i in line_data_list])
        # intr = torch.tensor([[self.focal, 0, self.raw_W / 2],
        #                      [0, self.focal, self.raw_H / 2],
        #                      [0, 0, 1]]).float()
        intr = torch.tensor([
                [cam_intrinsics[0][2], 0, cam_intrinsics[0][4]],
                [0, cam_intrinsics[0][3], cam_intrinsics[0][5]],
                [0, 0, 1]
            ]).float()
        # origin video's origin_size(1920,1440) -> extract frame (640,480)
        ori_size = (1920, 1440)
        size = (640, 480)
        intr[0,:] /= (ori_size[0] / size[0])
        intr[1, :] /= (ori_size[1] / size[1])  #resize 전 크기가 orgin_size 이기 때문에

        pose_raw = torch.tensor(self.list[idx],dtype=torch.float32)
        pose = self.parse_raw_camera(opt,pose_raw) #pose_raw (3,4)
        return intr,pose

    # [right, forward, up]
    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1]))) #camera frame change (3,4)  [[1., 0., 0., 0.],[0., -1., 0., 0.],[0., 0., -1., 0.]]
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])  # [right,up,back]->[right, down, forward] , pose_raw[:3]=pose_flip=(3,4),(3,4)
        pose = camera.pose.invert(pose)  #아마 c2w->w2c?
        return pose