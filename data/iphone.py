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
        self.raw_H,self.raw_W = 1080,1920
        super().__init__(opt,split)
        self.root = opt.data.root or "data/iphone"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}/iphone_train_val_images".format(self.path) if split != "test" else "{}/test".format(self.path)
        self.list = sorted(os.listdir(self.path_image), key=lambda f: int(f.split(".")[0]))  #이미지
        pose_fname = "{}/transforms_iphone.txt".format(self.path) if split != "test" else "{}/transforms_{}.txt".format(self.path, split)
        pose_file = os.path.join('./', pose_fname)
        assert os.path.isfile(pose_file), "pose info:{} not found".format(pose_file)
        with open(pose_file, "r") as f:  # frame.txt 읽어서
            cam_frame_lines = f.readlines()
        cam_pose = []  # r1x y z tx r2x y z ty r3x y z tz
        self.frames = []  # timestamp imagenum r1x y z tx r2x y z ty r3x y z tz
        for line in cam_frame_lines:
            line_data_list = line.split(' ')
            if len(line_data_list) == 0:
                continue
            self.frames.append(line_data_list)
            pose_raw = np.reshape(line_data_list[2:], (3, 4))
            cam_pose.append(pose_raw)
        cam_pose = np.array(cam_pose, dtype=float)
        self.cam_pose = cam_pose

        if split != "test" : #train,val
            # manually split train/val subsets
            num_val_split = int(len(self) * opt.data.val_ratio)  # len * 0.1
            self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]  # 전체에서 0.9 : 0.1 = train : test 비율
            self.cam_pose = self.cam_pose[:-num_val_split] if split == "train" else self.cam_pose[-num_val_split:]

        self.gt_pose = self.cam_pose

        self.opti_pose = self.cam_pose
        # for GT data(optitrack)
        # gt_pose_fname = "{}/opti_transforms_{}.txt".format(self.path,split)
        # gt_pose_file = os.path.join('./', gt_pose_fname)
        # if os.path.isfile(gt_pose_file): # gt file exist
        #     with open(gt_pose_file, "r") as f:  # frame.txt 읽어서
        #         cam_frame_lines = f.readlines()
        #     cam_gt_pose = []  # time r1x y z tx r2x y z ty r3x y z tz
        #     for line in cam_frame_lines:
        #         line_data_list = line.split(' ')
        #         if len(line_data_list) == 0:
        #             continue
        #         pose_raw = np.reshape(line_data_list[1:], (3, 4))
        #         cam_gt_pose.append(pose_raw)
        #     cam_gt_pose = np.array(cam_pose, dtype=float)
        #     self.opti_pose = cam_gt_pose
        # else: self.opti_pose = self.cam_pose


        if subset:
            self.list = self.list[:subset] # val 4개만
            self.cam_pose = self.cam_pose[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt, self.get_camera,
                                                  data_str="cameras")  # get_all_camera_poses 로 감
    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt): #기본 train할때 여기 접근해서 가져오고, data 로드할때 여기 접근
        if self.split == 'test':
            pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
            pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        else:   #train,val initial pose I
            pose = camera.pose(t=torch.zeros(len(self),3))  # TODO :Camera 초기 포즈
        return pose

    #get_all_gt_camera_poses
    def get_all_gt_camera_poses(self,opt): # optitrack pose load
        #여기 iphone pose 평가할때 gt 데이터 로드 위해(train,val,test)
        pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
        pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose

    #get_all_gt_camera_poses
    def get_all_optitrack_camera_poses(self,opt): # optitrack pose load
        #여기 iphone pose 평가할때 gt 데이터 로드 위해(train,val,test)
        pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
        pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose


    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,  #shape (3,4)
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx])
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
        # self.focal = self.raw_W*4.2/(12.8/2.55)
        # intr = torch.tensor([[self.focal,0,self.raw_W/2],
        #                      [0,self.focal,self.raw_H/2],
        #                      [0,0,1]]).float()
            # frame.txt -> cam_instrinsic
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

        if self.split == 'test':
            pose_raw = torch.tensor(self.cam_pose[idx],dtype=torch.float32)
            pose = self.parse_raw_camera(opt, pose_raw)
        else: pose = camera.pose(t=torch.zeros(3))
        return intr,pose

    # [right, forward, up]
    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose