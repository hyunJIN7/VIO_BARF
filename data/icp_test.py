import numpy as np
import os,sys,time
import torch
import icp
# import camera

"icp for ios_logger and optitrack"

# Constants
N = 10  # 데이터셋 크기
num_tests = 10  # 반복 테스트 계산 횟수
dim = 3  # 데이터 포인트 차원
noise_sigma = .01  # 노이즈 표준 편차
translation = .1  # 테스트셋 최대 이동 거리
rotation = .1  # 테스트셋 최대 회전 각

def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():
    # Generate a random dataset
    A = np.random.rand(N, dim) #(N,3)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation) #(3,3)
        B = np.dot(R, B.T).T #(N,3)

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp(dir,opt='train',arkit_line=[],opti_line=[]):
    # opti_transforms_train.txt vs transforms_train.txt
    # opti :  timestamp r11 r12 r13 x r21 r22 r23 y r31 r32 r33 z
    # arkit : timestamp imagename r11 r12 r13 x r21 r22 r23 y r31 r32 r33 z

    #TODO : shape check
    #TODO: A,B order check
    opti_raw_xyz = np.array([opti_line[:,4], opti_line[:,8], opti_line[:,12]])
    arkit_xyz = np.array([arkit_line[:,5], arkit_line[:,9], arkit_line[:,13]])

    total_time = 0
    final_RT = []
    for i in range(num_tests):
        # ICP 알고리즘 실행
        T, distances, iterations = icp.icp(opti_raw_xyz, arkit_xyz, tolerance=0.000001) #T(4,4)
        final_RT = T


        # 동차좌표 생성
        # C = np.ones((N, 4))
        # C[:, 0:3] = np.copy(opti_raw_xyz)
        # # 변환행렬 적용
        # C = np.dot(T, C.T).T

        print('distance: {:.3}'.format(np.mean(distances)))
        # assert np.mean(distances) < 6 * noise_sigma  # 평균 에러
        # assert np.allclose(T[0:3, 0:3].T, R, atol=6 * noise_sigma)  # T and R should be inverses
        # assert np.allclose(-T[0:3, 3], t, atol=6 * noise_sigma)  # T and t should be inverses

    # print('icp time: {:.3}'.format(total_time / num_tests))

    opti_raw_pose = np.reshape(opti_line[:,1:], (-1,3,4)) #(n,3,4)?
    # opti_pose = camera.to_hom(torch.from_numpy(opti_pose)) #(n,4,4)??
    #final_RT (4,4) make (n,4,4)?
    # opti_raw_pose = final_RT @ opti_raw_pose # dim???..
    # use camera.pose.compose

    opti_raw_pose = torch.Tensor(opti_raw_pose)
    final_RT = torch.Tensor(final_RT)

    opti_pose = []
    lines = []
    for i in range(len(opti_raw_pose)):
        cam_pose = opti_raw_pose[i] #(3,4)
        pose_new = compose_pair(final_RT[:3], cam_pose) #TODO : order check
        pose_new = torch.reshape( pose_new, (1,-1))
        opti_pose.append(pose_new)

        pose_new = pose_new.numpy()[0]
        line = np.concatenate( [[opti_line[i,0]], pose_new]).astype(np.str)
        # pose_new = pose_new.numpy()[0]
        # line.append(str(pose_new[0,a]) for a in range(12))
        # print(line)
        lines.append(' '.join(line) + '\n')
    opti_pose_file = os.path.join(dir, 'icp_opti_transforms_{}.txt'.format(opt))
    with open(opti_pose_file, 'w') as f:
        f.writelines(lines)
    return np.array(opti_pose)

    return opti_pose

def compose_pair(pose_a,pose_b): #pose_new,pose
    # pose_new(x) = pose_b o pose_a(x)
    R_a,t_a = pose_a[...,:3],pose_a[...,3:]
    R_b,t_b = pose_b[...,:3],pose_b[...,3:]
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)[...,0] #(3)
    t_new = torch.reshape(t_new,(3,1))
    pose_new = torch.empty(3, 4)
    pose_new[...,:3] = R_new
    pose_new[...,3:] = t_new
    return pose_new

if __name__ == "__main__":
    test_best_fit()
    test_icp()