import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import imageio
import json
import cv2
from transforms3d.quaternions import quat2mat
from skimage import img_as_ubyte

from bisect import bisect_left
from icp_test import *

"""
[right,up,back]
keframe select : every 30 frames for opti-track

code 깔끔하지 않음…
"""
# cd data 한 다음에 이 코드 실행해야하나봐 경로 이상해
# python process_arkit_data3.py --expname stair_llff01
# 다 실행한 이후엔 cd ../ 해주고
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./arkit/',
                        help='input data directory')

    #keyframe options
    parser.add_argument("--data_frame_interval", type=int, default=30,
                        help='key frame interval to sync with the opti-track')

    #data option
    parser.add_argument("--data_val_ratio", type=float, default=0.1,
                        help='ratio of sequence split for validation')

    # python process_arkit_data_frame3.py --expname piano_1
    # python process_arkit_data_frame3.py --expname opti_truck03

    # optitrack option
    parser.add_argument("--use_optitrack", type=bool, default=False)
    parser.add_argument("--opti_pose_fanme", type=str, default='opti_pose_truck02_996.txt',help='optitrack file name')
    # python process_arkit_data_frame3.py --expname opti_truck10 --use_optitrack=True --opti_pose_fanme=icp_opti_translation_0621_truck10.txt
    # python process_arkit_data_frame3.py --expname opti_truck_03 --use_optitrack=True --opti_pose_fanme=opti_pose_truck_03_96.txt
    return parser

def nearest(s,ts):
    # Given a presorted list of timestamps:  s = sorted(index)
    i = bisect_left(s, ts)
    return min(s[max(0, i-1): i+2], key=lambda t: abs(ts - t)) #이건 index아닌 값 출력
# def nearest(s,ts):
#     return min(s, key=lambda t: abs(ts - t)) #이건 index아닌 값 출력

def nearest_index(s,ts):
    # Given a presorted list of timestamps:  s = sorted(index)
    time_list = list(map(lambda t: abs(ts - t), s))
    return time_list.index(min(time_list))

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def extract_frames(video_path, out_folder, size):
    origin_size=[]
    """mp4 to image frame"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.resize(frame, size) #이미지 사이즈 변경에 따른 instrinsic 변화는 아래에 있음
        cv2.imwrite(os.path.join(out_folder, str(i).zfill(5) + '.jpg'), frame)
    return origin_size

#SyncedPose.txt 만드는
def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    """Load camera intrinsics"""  # frane.txt -> camera intrinsics
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f:  # frame.txt 읽어서
        cam_intrinsic_lines = f.readlines()

    cam_intrinsics = []
    for line in cam_intrinsic_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_intrinsics.append([float(i) for i in line_data_list])
        # frame.txt -> cam_instrinsic
    K = np.array([
            [cam_intrinsics[0][2], 0, cam_intrinsics[0][4]],
            [0, cam_intrinsics[0][3], cam_intrinsics[0][5]],
            [0, 0, 1]
        ])

    """load camera keyframe_poses"""  # ARPose.txt -> camera pose  gt
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()
    cam_poses = []
    for line in cam_pose_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])

    """ outputfile로 syncpose 맞춰서 내보냄  """
    lines = []
    ip = 0
    length = len(cam_poses)

    for i in range(len(cam_intrinsics)):
        while ip + 1 < length and abs(cam_poses[ip + 1][0] - cam_intrinsics[i][0]) < abs(
                cam_poses[ip][0] - cam_intrinsics[i][0]):
            ip += 1
        cam_pose = cam_poses[ip][:4] + cam_poses[ip][5:] + [cam_poses[ip][4]]
        line = []
        line.append(str(cam_intrinsics[i][0])) # timestamp
        line.append(str(i).zfill(5))  # timestamp
        for p in cam_pose[1:] : line.append(str(p))
        #line.append(str(a) for a in cam_pose[1:])
        #line = [str(a) for a in cam_pose] #time,name,tx,ty,tz,qw,qx,qy,qz #TODO : timestamp.....
        lines.append(' '.join(line) + '\n')

    dirname = os.path.dirname(out_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_file, 'w') as f:
        f.writelines(lines)
    return K

def load_camera_pose(cam_pose_dir): # SyncedPose.txt
    if cam_pose_dir is not None and os.path.isfile(cam_pose_dir):
        pass
    else:
        raise FileNotFoundError("Given camera pose dir:{} not found"
                                .format(cam_pose_dir))

    pose = []         #[right,forward,up]
    pose_raw = [] # [right,up,back]
    timestamp_name = []
    def process(line_data_list):   #syncedpose.txt  : timestamp imagenum(string) tx ty tz(m) qx qy qz qw
        line_data = np.array(line_data_list, dtype=float)
        timestamp_name.append(line_data[:2]) # timestamp, name
        # fid = line_data_list[0] #0부터
        trans = line_data[2:5]
        quat = line_data[5:]
        rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
                            # 여기선 (w,x,y,z) 순 인듯
        trans_mat = np.zeros([3, 4])
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans
        trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
        pose_raw.append(trans_mat)

        #[right,forward,up]
        rot_mat = rot_mat.dot(np.array([  #axis flip..?
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]))
        rot_mat = rotx(np.pi / 2) @ rot_mat #3D Rotation about the x-axis.
        trans = rotx(np.pi / 2) @ trans
        trans_mat = np.zeros([3, 4])
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans
        trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
        pose.append(trans_mat)

    with open(cam_pose_dir, "r") as f:
        cam_pose_lines = f.readlines()
    for cam_line in cam_pose_lines:
        line_data_list = cam_line.split(" ")
        if len(line_data_list) == 0:
            continue
        process(line_data_list)

    return pose_raw,pose,timestamp_name


def process_arkit_data(args,ori_size=(1920, 1440), size=(640, 480)):
    basedir = os.path.join(args.basedir,args.expname)

    # print('Extract images from video...')
    video_path = os.path.join(basedir, 'Frames.m4v')
    image_path = os.path.join(basedir, 'images')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        extract_frames(video_path, out_folder=image_path, size=size) #조건문 안으로 넣음

    # make SyncedPose
    print('Load intrinsics and extrinsics')
    K = sync_intrinsics_and_poses(os.path.join(basedir, 'Frames.txt'), os.path.join(basedir, 'ARposes.txt'),
                            os.path.join(basedir, 'SyncedPoses.txt')) #imagenum(string) tx ty tz(m) qx qy qz qw
    # K[0,:] /= (ori_size[0] / size[0])
    # K[1, :] /= (ori_size[1] / size[1])  #resize 전 크기가 orgin_size 이기 때문에

    #quat -> rot
    all_raw_cam_pose,all_cam_pose,sync_timestamp_name = load_camera_pose(os.path.join(basedir, 'SyncedPoses.txt'))

    """Keyframes selection
        arkit 30Hz 라 가정할때 
        30개당 하나씩 추출. 0번째 추출.
    """
    all_ids = [i for i in range(len(all_cam_pose)) if i%args.data_frame_interval == 0]

    """final select image,keyframe_poses  for train,val data"""
    keyframe_imgs = []
    keyframe_poses = []
    keyframe_timestamp_name = []
    for i in all_ids:
        image_file_name = os.path.join(image_path, str(i).zfill(5) + '.jpg')
        keyframe_imgs.append(imageio.imread(image_file_name))
        keyframe_poses.append(all_raw_cam_pose[i])
        keyframe_timestamp_name.append(sync_timestamp_name[i])
    keyframe_imgs = (np.array(keyframe_imgs) / 255.).astype(np.float32)
    keyframe_poses = np.array(keyframe_poses).astype(np.float32)
    keyframe_timestamp_name = np.array(keyframe_timestamp_name)


    """train, val, test  ver2"""
    # n = keyframe_poses.shape[0]  # count of image
    # num_val_split = (int)(n * args.data_val_ratio)
    # num_of_data = [100,104,134] # 100 : 4 : 30
    # train_indexs = np.linspace(0, n, n, endpoint=False, dtype=int)[:num_of_data[0]]
    # val_indexs = np.linspace(0, n, n, endpoint=False, dtype=int)[num_of_data[0]:num_of_data[1]]
    # test_indexs = np.linspace(0, n, n, endpoint=False, dtype=int)[num_of_data[1]:num_of_data[2]] #np.random.choice(len(all_raw_cam_pose) , int(n*0.25), replace=False) #키프레임셀렉에서말고 전체 싱크 맞춘거에서 테스트 데이터 뽑아,비복원추출
    # # test_indexs.sort()
    # iphone_train_val = np.concatenate((train_indexs,val_indexs))
    # iphone_train_val.sort()

    """train, val, test"""
    n = keyframe_poses.shape[0]  # count of image
    num_val_split = (int)(n * args.data_val_ratio)
    train_indexs = np.linspace(0, n, n, endpoint=False, dtype=int)[:-num_val_split]
    val_indexs = np.linspace(0, n, n, endpoint=False, dtype=int)[-num_val_split:]
    test_indexs = np.random.choice(len(all_raw_cam_pose) , int(n*0.25), replace=False) #키프레임셀렉에서말고 전체 싱크 맞춘거에서 테스트 데이터 뽑아,비복원추출
    test_indexs.sort()
    iphone_train_val = np.concatenate((train_indexs,val_indexs))
    iphone_train_val.sort()



    # transform.txt 파일에 index 있긴하지만 nerf-- format 따르기 위해 따로 index.txt 만들어주기 위해
    def save_image_index(opt='train', indexs=[]):
        index_fild = os.path.join(basedir, '{}_ids.txt'.format(opt))
        lines = []
        # lines.append(' '.join([str(indexs[i])]) + '\n' for i in range(len(indexs)))  # error
        for i in range(len(indexs)):
            line = [str(indexs[i])]
            lines.append(' '.join(line) + '\n')
        with open(index_fild, 'w') as f:
            f.writelines(lines)
    save_image_index('train',train_indexs)
    save_image_index('val', val_indexs)
    save_image_index('test', test_indexs)

    """
        final select image,keyframe_poses for test data
        test pose는 select 된 pose가 아닌 sync 맞춘 전체 pose 데이터에서 추출하는 것이기 때문
    """
    test_imgs = []
    test_poses = []
    test_timestamp_name = []
    for i in test_indexs:
        image_file_name = os.path.join(image_path, str(i).zfill(5) + '.jpg')
        test_imgs.append(imageio.imread(image_file_name))
        test_poses.append(all_raw_cam_pose[i])
        test_timestamp_name.append(sync_timestamp_name[i])
    test_imgs = (np.array(test_imgs) / 255.).astype(np.float32)
    test_poses = np.array(test_poses).astype(np.float32)
    test_timestamp_name = np.array(test_timestamp_name)
    print('train : {0} , val : {1} , test : {2}'.format(train_indexs.shape[0],val_indexs.shape[0],test_indexs.shape[0]))


    #for iphone train data
    iphone_image_dir = os.path.join(basedir, 'iphone_train_val_images')
    if not os.path.exists(iphone_image_dir):
        os.mkdir(iphone_image_dir)
    iphone_pose_fanme = os.path.join(basedir , 'transforms_iphone.txt')
    iphone_poses_str = []  # for barf identity experiment

    # select pose,image 파일 저장
    def save_keyframe_data(dir, opt='train', index=[] ,images=[], pose=[],all_cam_timestamp_name_pose=[]):
        image_dir = os.path.join(dir, opt)
        pose_file = os.path.join(dir, 'transforms_{}.txt'.format(opt)) #time imagename pose
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        lines = []
        arkit_pose = []
        for i in range(len(index)):
            line = []
            pose_line = []
            imageio.imwrite('{}/{}.jpg'.format(image_dir,str(int(all_cam_timestamp_name_pose[i,1]) ).zfill(5)), img_as_ubyte(images[i]))
            if opt != 'test': # for iphone
                imageio.imwrite('{}/{}.jpg'.format(iphone_image_dir, str(int(all_cam_timestamp_name_pose[i, 1])).zfill(5)),
                                img_as_ubyte(images[i]))
            line.append(str(all_cam_timestamp_name_pose[i,0])) # timestamp
            line.append(opt+'/' + str(int(all_cam_timestamp_name_pose[i,1]) ).zfill(5) ) # image name
            pose_line.append(all_cam_timestamp_name_pose[i,0])
            pose_line.append(all_cam_timestamp_name_pose[i,1])


            for j in range(3):
                for k in range(4) :
                    line.append(str(pose[i][j][k]))
                    pose_line.append(pose[i][j][k])
            #line =np.concatenate((pose[i][0,:] ,pose[i][1,:] , pose[i][2,:]) , axis=0  )        #pose[i][0,:3] + pose[i][1,:3] + pose[i][2,:3] \
            lines.append(' '.join(line) + '\n') # (3x4)shape이 row 한줄로 이어 붙임.
            arkit_pose.append(pose_line)

            if opt != 'test': # for iphone
                iphone_poses_str.append(' '.join(line) + '\n')

        with open(pose_file, 'w') as f:
            f.writelines(lines)

        return np.array(arkit_pose)

    arkit_pose_train = save_keyframe_data(basedir,'train',
                       train_indexs,
                       keyframe_imgs[train_indexs],
                       keyframe_poses[train_indexs],
                       keyframe_timestamp_name[train_indexs]);

    arkit_pose_val = save_keyframe_data(basedir,'val',
                       val_indexs,
                       keyframe_imgs[val_indexs],
                       keyframe_poses[val_indexs],
                       keyframe_timestamp_name[val_indexs]);

    arkit_pose_test = save_keyframe_data(basedir,'test',
                       test_indexs,
                       test_imgs,
                       test_poses,
                       test_timestamp_name);

    #for transforms_iphone.txt
    with open(iphone_pose_fanme, 'w') as f:
        f.writelines(iphone_poses_str)

    # optitrack data 있다면 sync 맞춘 optitrack 포즈 저장
    #train,val,test 에 대해서 다 찾아야해.
    if args.use_optitrack :
        # optitrack file 읽어오기
        opti_pose_file= os.path.join(basedir, args.opti_pose_fanme)
        assert os.path.isfile(opti_pose_file), "camera info:{} not found".format(opti_pose_file)
        with open(opti_pose_file, "r") as f:
            opti_lines_list = f.readlines()
        opti_lines = [] # optitrack data
        for line in opti_lines_list:
            line_data_list = line.split(' ')
            if len(line_data_list) == 0:
                continue
            opti_lines.append([float(i) for i in line_data_list])

        opti_raw_pose_train = sync_arkit_with_optitrack(basedir,'train',train_indexs , keyframe_timestamp_name[train_indexs][:,0], opti_lines)
        opti_raw_pose_val = sync_arkit_with_optitrack(basedir, 'val', val_indexs, keyframe_timestamp_name[val_indexs][:, 0], opti_lines)
        opti_raw_pose_test = sync_arkit_with_optitrack(basedir, 'test', test_indexs, test_timestamp_name[:, 0], opti_lines)

        #ICP
        # opti_pose_train = test_icp(basedir,'train',arkit_pose_train,opti_raw_pose_train)
        # opti_pose_val = test_icp(basedir, 'val', arkit_pose_train, opti_raw_pose_train)
        # opti_pose_test = test_icp(basedir, 'test', arkit_pose_test, opti_raw_pose_test)
        # #save opti_pose txt

def sync_arkit_with_optitrack(dir,opt='train',index=[],arkit_timestamp=[] , opti_lines=[]):
    """
        optitrack 과 arkit sync 맞추는 코드
        train,val,test 에 대해서 다 찾아야해.
        그 train,val,test 각각의 인덱스 있으니까 그 인덱스의 요소와
        str(all_cam_timestamp_name_pose[i,0] 이 값이랑 optitrack 값이랑 비교하면 되겠다.
    """
    lines = []
    opti_pose = []
    opti_lines = np.array(opti_lines)
    for i in range(len(index)):
        min_index = nearest_index(opti_lines[:,0], arkit_timestamp[i])
        opti_pose.append(opti_lines[min_index])
        line = [str(a) for a in opti_lines[min_index]]
        lines.append(' '.join(line)+'\n')
    opti_pose_file = os.path.join(dir, 'opti_transforms_{}.txt'.format(opt))
    with open(opti_pose_file, 'w') as f:
        f.writelines(lines)
    return np.array(opti_pose)

# cd data 한 다음에 이 코드 실행해야하나봐 경로 이상해
# python process_arkit_data_frame3.py --expname half_box
# 다 실행한 이후엔 cd ../ 해주고
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    process_arkit_data(args)
