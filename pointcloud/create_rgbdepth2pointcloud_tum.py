import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    # parser.add_argument("--i_video",   type=int, default=50000,
    #                     help='frequency of render_poses video saving')
    # parser.add_argument("--half_res", action='store_true',
    #                     help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--basedir", type=str, default='./pointcloud',
                        help='input data directory')
    parser.add_argument("--datadir", type=str, default='',
                        help='input data directory')
    parser.add_argument("--expname", type=str, default='/fern_test_origin',
                        help='experiment name')

    return parser


# def draw_image():
    # plt.subplot(1, 2, 1)
    # plt.title('TUM grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('TUM depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()


def load_data(args):
    # basedir = os.path.join(args.datadir,args.expname)
    # color_dir = os.path.join(args.datadir, 'testset_200000')
    # depth_dir = os.path.join(args.datadir, 'testset_depth_200000')


    #test
    color_dir = os.path.join(args.basedir, 'color')
    depth_dir = os.path.join(args.basedir, 'depth')


    print("Read TUM dataset")
    color_list = os.listdir(color_dir)
    depth_list = os.listdir(depth_dir)

    all_color = []
    all_depth = []
    all_rgbd = []
    for i in range(len(color_list)):
        color_raw = o3d.io.read_image(os.path.join(color_dir, color_list[i]))
        depth_raw = o3d.io.read_image(os.path.join(depth_dir, depth_list[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)

        all_color.append(color_raw)
        all_depth.append(depth_raw)
        all_rgbd.append(rgbd_image)
        print(rgbd_image)
    #draw_image(all_rgbd,all_depth)
    return all_color, all_depth, all_rgbd

def draw_pcd(args):

    all_color,all_depth,all_rgbd = load_data(args)

    all_pcd= o3d.geometry.PointCloud()

    # for rgbd_image in enumerate(all_rgbd):
    for i in range(len(all_rgbd)):

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            all_rgbd[i],
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
        all_pcd += pcd
    all_pcd_down = all_pcd.voxel_down_sample(voxel_size=0.0005)
    #all_pcd_down = all_pcd

    o3d.visualization.draw_geometries([all_pcd_down])#, zoom=0.35) zoom error


#python pointcloud/create_rgbdepth2pointcloud.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    draw_pcd(args)