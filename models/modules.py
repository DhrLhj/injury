import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests

from scipy.spatial.distance import cdist



#--------------------------------------------------------------------------
# 数据可视化
#--------------------------------------------------------------------------
def plotgesture(arr_total):
    """
    手势绘制
    """
    list_c = ["r*","g-","g*","r-","b*","b-"]
    assert arr_total.shape[1] <= len(list_c) 
    plt.figure()
    s = arr_total.shape[1]
    for i in range(s):
        plt.plot(arr_total[0,i,0],arr_total[0,i,1],"yo")# 左肘
        plt.plot(arr_total[1:,i,0],arr_total[1:,i,1],list_c[i])# 左肘
    plt.show()

#------------------------------------------------------------------------------------------
# 数据处理
#------------------------------------------------------------------------------------------



def coord_norm(arr_coords):
    """
    手、肘相对于肩膀中心的距离，与躯干的长度的比值；
    输入：
    [[-0.0944298  0.113541  -1.57483  ] 左肘
     [-0.0508029 -0.118521  -1.50818  ] 左手
     [ 0.383744   0.12448   -1.52399  ] 右肘
     [ 0.397039  -0.108943  -1.44219  ] 右手
     [ 0.142538   0.434921  -1.54355  ] 肩膀中心
     [-0.153653  -0.206411  -1.56354  ]] 躯干中心
    返回：
    numpy array: (4,2)
    手、肘的横纵坐标
    """
    arr_norm = (arr_coords[:2,:2]-arr_coords[4,:2])/(2*distance(arr_coords[4,:2],arr_coords[5,:2]))
#     arr_diff = arr_norm[1:] - arr_norm[:-1]
    return  arr_norm
#     return  (arr_coords[:4]-arr_coords[5])

def distance(coorda,coordb):
    """
    计算两点欧式距离
    """
    return np.sqrt(((coorda-coordb)**2).sum())

def vDistance(v1,v2):
    """
    计算四个关键点的特征向量的总的距离差
    @param v1:
    @param v2:
    v1,v2为4个特征点的坐标组成的数组，shape:(4,3) 
    """
    return np.sqrt(((v1-v2)**2).sum(1)).sum()

def vDistance2(v1,v2):
    """
    计算两手的两个关键点的特征向量的总的距离差
    @param v1:
    @param v2:
    v1,v2为4个特征点的坐标组成的数组，shape:(4,3) 
    """
    return (np.sqrt(((v1-v2)**2).sum(1))*np.array([0.8,1.2])).sum()

def normalization(coord,neck,center):
    """
    坐标中心化归一化
    @param coord:
    @param neck:
    @param center:
    @return coord
    """
    # neck中心到颈椎中心的距离
    return (coord-center)/distance(neck,center)


if __name__ == "__main__":
#     arr_coords = get_coordinates("../data_k2/右手上滑.csv")
#     # 将坐标中心归一化
#     print(coord_norm(arr_coords[0]))
    
#     print(getTempGestures("../GestureTemplate").shap
    pass