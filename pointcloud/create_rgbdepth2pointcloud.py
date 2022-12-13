import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
rgb,depth,intrinsic,pose 필요 

ply 파일로 저장시켜서 mesh 만들어 
"""
# python pointcloud/create_rgbdepth2pointcloud.py --extrinsic transforms_test.txt --expname lego --depth_scale 1000????
# python pointcloud/create_rgbdepth2pointcloud.py --extrinsic extrinsic.txt --expname llff_lab01 --ptd_down_sample False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # parser.add_argument("--half_res", action='store_true',
    #                     help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--basedir", type=str, default='./pointcloud',
                        help='input data directory')
    # parser.add_argument("--datadir", type=str, default='',
    #                     help='input data directory')
    parser.add_argument("--expname", type=str, default='llff_lab01',
                        help='experiment name')
    parser.add_argument("--extrinsic", type=str, default='extrinsic.txt',
                        help='extrinisc.txt name')

    #depth
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help='depth_scale')
    parser.add_argument("--ptd_down_sample", type=bool, default=True,
                        help='point cloud down sampling')
    return parser

def draw_image(rgbd):
    plt.subplot(1, 2, 1)
    plt.title('rgb image')
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd.depth)
    plt.show()

def load_iamge_data(args):
    # color_dir = os.path.join(args.basedir, args.expname,'color')
    # depth_dir = os.path.join(args.basedir, args.expname,'depth')

    color_dir = os.path.join('pointcloud','color')
    depth_dir = os.path.join('pointcloud','depth')

    print("Read dataset")
    color_list = os.listdir(color_dir)
    depth_list = os.listdir(depth_dir)

    all_color = []
    all_depth = []
    all_rgbd = []
    for i in range(len(color_list)):
        color_raw = o3d.io.read_image(os.path.join(color_dir, color_list[i]))
        depth_raw = o3d.io.read_image(os.path.join(depth_dir, depth_list[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=args.depth_scale)

        all_color.append(color_raw)
        all_depth.append(depth_raw)
        all_rgbd.append(rgbd_image)
        print(rgbd_image)
    draw_image(all_rgbd[0])
    return all_color, all_depth, all_rgbd

# 나중에 파일 로드해서 읽어오는 방식으로 수정
def load_intrinsic(args,raw_intr):

    ori_size = (1920, 1440)
    size = (640, 480)
    raw_intr[0] /= (ori_size[0] / size[0])
    raw_intr[2] /= (ori_size[0] / size[0])
    raw_intr[1] /= (ori_size[1] / size[1])
    raw_intr[3] /= (ori_size[1] / size[1])
    intr = [[raw_intr[0], 0, raw_intr[2]],
            [0, raw_intr[1], raw_intr[3]],
            [0, 0, 1]]
    return intr

def load_extrinsic(args):
    extrinsic_file = os.path.join(args.basedir, args.expname,args.extrinsic)
    assert os.path.isfile(extrinsic_file), "pose info:{} not found".format(extrinsic_file)
    with open(extrinsic_file, "r") as f:  # frame.txt 읽어서
        pose_lines = f.readlines()

    cam_extrinsic = []
    for line in pose_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        raw_pose = np.reshape(line_data_list, (3,4))
        raw_pose_hom = np.vstack((raw_pose, np.array([0,0,0,1])))
        # raw_pose_hom = np.concatenate([raw_pose, np.ones_like(raw_pose[...,:1])], axis=-1) #, axis=0
        cam_extrinsic.append(raw_pose_hom) # append 하면 리스트로 되던뎅...
    return cam_extrinsic

def draw_pcd(args):

    all_color,all_depth,all_rgbd = load_iamge_data(args)
    all_pcd= o3d.geometry.PointCloud()

    #TODO:나중에 여기도 처리
    raw_intr = [1510.745728,1510.745728,964.076294,722.721863] #fx fy cx cy
    ori_size = (1920, 1440)
    size = (640, 480)
    intr = load_intrinsic(args,raw_intr)
    # extrin = load_extrinsic(args)

    # for rgbd_image in enumerate(all_rgbd):
    for i in range(len(all_rgbd)):
        cam_intr = o3d.camera.PinholeCameraIntrinsic(size[0], size[1], raw_intr[0], raw_intr[1], raw_intr[2],raw_intr[3])
        cam_intr.intrinsic_matrix = intr
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = cam_intr
        # cam.extrinsic = extrin[i]   #np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            all_rgbd[i],
            cam.intrinsic,
            # cam.extrinsic
        )
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
        all_pcd += pcd
    all_pcd_down = all_pcd.voxel_down_sample(voxel_size=0.0005) if args.ptd_down_sample else all_pcd
    o3d.visualization.draw_geometries([all_pcd_down]) #,zoom=0.35) zoom error

# python pointcloud/create_rgbdepth2pointcloud.py --extrinsic transforms_test.txt --expname llff_main_computers_03
# python pointcloud/create_rgbdepth2pointcloud.py --expname arkit_llff_main_computers02_novel --ptd_down_sample False
# python pointcloud/create_rgbdepth2pointcloud.py --expname arkit_llff_main_computers02_test
# python pointcloud/create_rgbdepth2pointcloud.py --expname iphone_llff_main_computers02_novel
# python pointcloud/create_rgbdepth2pointcloud.py --expname iphone_llff_main_computers02_test

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    draw_pcd(args)