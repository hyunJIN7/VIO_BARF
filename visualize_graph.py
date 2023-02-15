import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,make_interp_spline

import numpy as np
import matplotlib.ticker as ticker
from scipy import signal
import numpy as np


def scatter_point(data1,data2=None,data3=None):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data1[:,0],data1[:,1],data1[:,2],color='r',alpha = 0.5)
    if data2 is not None :
        ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2],color='g' ,alpha=0.5)
    if data3 is not None :
        ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2],color='b' ,alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_line(data1, data2=None):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    N = data1.shape[0]
    for i in range(N):
        ax.plot([0,data1[i,0]],[0,data1[i,1]], [0,data1[i,2]], color='r', linewidth=0.5, alpha=0.5)

    if data2 is not None:
        N = data2.shape[0]
        for i in range(N):
            ax.plot([0, data2[i, 0]], [0, data2[i, 1]], [0, data2[i, 2]], color='b', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # png_fname = "{}/{}.png".format(path,ep)
    # plt.savefig(png_fname,dpi=75)
    # clean up
    # plt.clf()



flg2 = plt.figure()      # 차트 플롯 생성
chart = flg2.add_subplot(1, 1, 1)    # 행, 열, 위치

start = -10
time = 100
interval= 0.1
# data 생성
data1 = np.random.randn(time)/15 + 5*np.sin(np.arange(0,time/4,0.25))  +  np.arange(-time/2,time/2) * np.arange(-time/2,time/2)/ 80   # 난수 np.random.randn(time)*0.2 +
data2 = np.random.randn(time*100).cumsum() # +  np.sin(np.arange(0,time,0.01))  # 난수 -> 누적 합

# y = np.arange(0,time,interval)
# # data3 = np.sin(np.arange(start,time,interval))*10 + y*y# + data1*0
# data4 = np.sin(np.arange(0,time,interval)) + 3*np.cos(np.arange(0,time/interval)) + y

data = data1

# 계단형 차트
# chart.plot(data4[100:150], label='step',color='r')#, drawstyle='steps', color='r')
# 선 스타일 차트
c=(0.6,0.6,0.6)
chart.plot(data, label='line',c=c,linewidth=7)
plt.title("multi chart draw")      # 차트 제목
plt.xlabel('stage')                 # x축
plt.ylabel('random number')    # y축
plt.legend(loc='best')            # 범례
plt.show()
