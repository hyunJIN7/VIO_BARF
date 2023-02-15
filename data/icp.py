import numpy as np
from sklearn.neighbors import NearestNeighbors

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    주어짐 점군 A, B에 대해 정합 행렬을 계산해 리턴함.
    Input:
        A: numpy 형태 Nxm 행렬. 소스(Src) mD points
        B: numpy 형태 Nxm 행렬. 대상(Dst) mD points
        init_pose: (m+1)x(m+1) 동차좌표계(homogeneous) 변환행렬
        max_iterations: 알고리즘 계산 중지 탈출 횟수
        tolerance: 수렴 허용치 기준
    Output:
        T: 최종 동차좌표계 변환 행렬. maps A on to B
        distances: 가장 가까운 이웃점 간 유클리드 오차 거리
        i: 수렴 반복 횟수
    '''

    assert A.shape == B.shape

    # 차원 획득
    m = A.shape[1]

    # 동차 좌표계 행렬을 만들고, 점군 자료를 추가
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # src 점군에 초기 자세 적용
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):  # 정합될때까지 반복 계산
        # 소스와 목적 점군 간에 가장 근처 이웃점 탐색. 계산량이 많음.
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # 소스 점군에서 대상 점군으로 정합 시 필요한 변환행렬 계산
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # 소스 점군에 변환행렬 적용해 좌표 갱신
        src = np.dot(T, src)

        # 에러값 계산
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:  # 허용치보다 에러 작으면 탈출
            break
        prev_error = mean_error

    # 변환행렬 계산
    T, _, _ = best_fit_transform(A, src[:m, :].T)
def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    주어짐 점군 A, B에 대해 정합 행렬을 계산해 리턴함.
    Input:
        A: numpy 형태 Nxm 행렬. 소스(Src) mD points
        B: numpy 형태 Nxm 행렬. 대상(Dst) mD points
        init_pose: (m+1)x(m+1) 동차좌표계(homogeneous) 변환행렬
        max_iterations: 알고리즘 계산 중지 탈출 횟수
        tolerance: 수렴 허용치 기준
    Output:
        T: 최종 동차좌표계 변환 행렬. maps A on to B
        distances: 가장 가까운 이웃점 간 유클리드 오차 거리
        i: 수렴 반복 횟수
    '''

    assert A.shape == B.shape

    # 차원 획득
    m = A.shape[1]

    # 동차 좌표계 행렬을 만들고, 점군 자료를 추가
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # src 점군에 초기 자세 적용
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):  # 정합될때까지 반복 계산
        # 소스와 목적 점군 간에 가장 근처 이웃점 탐색. 계산량이 많음.
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # 소스 점군에서 대상 점군으로 정합 시 필요한 변환행렬 계산
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # 소스 점군에 변환행렬 적용해 좌표 갱신
        src = np.dot(T, src)

        # 에러값 계산
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:  # 허용치보다 에러 작으면 탈출
            break
    return T, distances, i


def nearest_neighbor(src, dst):
    '''
    소스와 목적 점군에 대한 가장 이웃한 점 계산
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: 가장 가까운 이웃점간 유클리드 거리
        indices: 목적 점군에 대한 가장 가까웃 이웃점의 인덱스들
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    '''
    m 공간 차원에서 점군 A에서 B로 맵핑을 위한 최소자승법 변환행렬 계산
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) 동차좌표 변환행렬. maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # 차원 얻기
    m = A.shape[1]

    # 각 점군 중심점 및 중심 편차 계산
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A  # A행렬 편차
    BB = B - centroid_B  # B행렬 편차

    # SVD이용한 회전 행렬 계산
    H = np.dot(AA.T, BB)  # 분산
    U, S, Vt = np.linalg.svd(H)  # SVD 계산
    R = np.dot(Vt.T, U.T)  # 회전 행렬

    # 반사된 경우 행렬 계산
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 이동 행렬 계산
    t = centroid_B.T - np.dot(R, centroid_A.T)  #

    # 동차 변환행렬 계산
    T = np.identity(m + 1)
    T[:m, :m] = R  # 회전 행렬
    T[:m, m] = t  # 이동 행렬

    return T, R, t
