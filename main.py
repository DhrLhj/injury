import os
import time
import sys
import glob
import math
import numpy as np
from typing import List

from models.dtw import dtw
from models.modules import coord_norm,distance,vDistance,plotgesture,vDistance2
from util.dataloader import get_k2_http,get_templates,get_coordinates,get_pose, pose_estimation
from util.winkey import key_input,key_test,mouse_move_coords,getdim
from config.config import list_key_office


def get_k2_data(arr_coords:np.array,i:int,lengthTemp:int)->np.array:
    """
    读取保存的坐标数据模拟实时生成的数据：
    根据循环的次数，读取一定长度的节点数据
    """
    assert (i+lengthTemp)<arr_coords.shape[0]
    return arr_coords[i:i+lengthTemp]

def cal_dtw(arr_temp:np.array,ar_coords:np.array,arr_coords_last:np.array, arr_norm)->int:
    """
    输入模板序列和实时数据序列，计算dtw值，处理后，返回匹配结果
    几种特殊情况的处理：
    **静止手势**：计算序列中各帧是否移动，如果所有帧都未移动，则判定为静止；
    **无手势**：
        (1)计算出的dtw结果都大于某个值；没有这种情况
        (2)dtw有符合要求的结果，针对每种情况进行判断；
            -向右滑：右手手肘和手部的X距离差；
            -向左滑：右手手和手部的X距离差
            -向上滑：Y轴距离差：
            -向下滑：Y轴距离差：
            
    return :
        -2 无手势：连续左右手坐标未移动
    """
    # 计算当前序列中每一帧对应的移动距离
    arr_coords_last = arr_coords-0.1
    # 计算当前序列中每一帧对应的移动距离
    arr_coords_last[0:-1] = arr_coords_last[1:]
    arr_coords_last[-1] = arr_coords[0]
    arr_dlh = np.sqrt(((arr_coords[:,1,:2]-arr_coords_last[:,1,:2])**2).sum(axis=1))
    arr_drh = np.sqrt(((arr_coords[:,3,:2]-arr_coords_last[:,3,:2])**2).sum(axis=1))
    
    # 移动距离判断
    # print(arr_dlh,arr_drh)
    isGestureAvailable = sum([(0 if ((d[0]<GESTURE_MIN_THRESHOLD) & (d[1]<GESTURE_MIN_THRESHOLD) )  else 1 )  for d in zip(arr_dlh,arr_drh)])
    if not isGestureAvailable: # 静止手势，跳过处理
        return -2,10
    # 移动距离判断
    
    # 右手，左手移动距离计算
    dlh = arr_dlh[-1] ### 左手移动距离
    drh = arr_drh[-1] ### 右手移动距离
    dlr = distance(arr_coords[-1,1],arr_coords[-1,3]) ### 左右手最后时刻相对距离
       
    # 计算 最后一个帧与第一帧的距离
    lw = b * (ar_coords[-1,1,:2]-ar_coords[0,1,:2]) #左手 最后的
    rw = b * (ar_coords[-1,3,:2]-ar_coords[0,3,:2]) #右手
    wb = [lw,rw]
    
#     bm = np.array([
#         np.exp(-1*rw[1])*np.exp(7*b*dlh),
#         np.exp(rw[1])*np.exp(7*b*dlh), 
#         np.exp(-1*rw[0])*np.exp(7*b*dlh), 
#         np.exp(rw[0])*np.exp(7*b*dlh), 
#         np.exp(lw[0]),
#         1,
#         np.exp(-1*lw[0])*np.exp(-1*lw[1]),
#         1,
#         1,
#         1,
#     ])
    
    bm = np.array([1]*10)
    list_dtw = []
    for t in range(arr_temp.shape[0]):
#         print(ar_coords.shape,arr_temp.shape)
        res,cost_matrix,acc_cost_matrix,path = dtw(ar_coords[:,:,:2],arr_temp[t][:,:,:2],vDistance2,w=4)
        list_dtw.append(res)

    arr_dtw = np.array(list_dtw)
    arr_res = bm[:arr_temp.shape[0]]*arr_dtw
#     arr_res = arr_dtw
    minIndex = np.argmin(arr_res)
    minRes = arr_res[minIndex]
    # 剔除匹配较大的值
#     if minRes > 5:
#         return -3, minRes
#     # 向上的约束
    
#     return minIndex, minRes
    print(f'#{minIndex}',arr_norm[0,1],arr_norm[0,3],arr_norm[-1,1],arr_norm[-1,3])
    
    # 剔除匹配较大的值
    if minRes > 10:
        return -3, minRes
    # 向上的约束
    elif minIndex == 0 and arr_norm[-1,3,1] > -0.3:  # 向上手势，右手必须在肩部以上(越下值越大)
        return -3, minRes
    elif minIndex == 0 and (arr_norm[-1,3,0] > 0.5 or arr_norm[-1,3,0]<0):  # 向上手势，右手向右的波动不能超过躯干的一半（大约大臂的长度）
        return -3, minRes
    # 向下的约束
    elif minIndex == 1 and arr_norm[-1,3,1] < 0.5:  # 向下手势，右手必须在肩部以下(越下值越大)
        return -3, minRes
    elif minIndex == 1 and (arr_norm[-1,3,0] > 0.6 or arr_norm[-1,3,0]<0.2) :  # 向下手势，右手向右的波动不能超过躯干的一半（大约大臂的长度）
        return -3, minRes
    elif minIndex == 1: # 向下手势，必须起始帧的右手的Y轴移动超过一半的躯干距离
        if arr_norm[-1,3,1] - arr_norm[0, 3, 1] < 0.4:
            return -3, minRes
    # 向左约束
    elif minIndex == 2 and arr_norm[-1,3,0] > 0:  # 向左手势，右手必须在头部左侧(越右值越大)
        return -3, minRes
    elif minIndex == 2 and (arr_norm[-1,3,1] > 0.2 or arr_norm[-1,3,1] < -0.2) :  # 向左手势，右手上下波动不能超过躯干的一半（大约大臂的长度）
        return -3, minRes
    # 向右约束
    elif minIndex == 3 and arr_norm[-1,3,0] < 0.5:  # 向右手势，右手必须在躯干一半值以外(越右值越大)
        return -3, minRes
    elif minIndex == 3 and (arr_norm[-1,3,1] > 0.2 or arr_norm[-1,3,1] < -0.2) :  # 向右手势，右手上下波动不能超过躯干的一半（大约大臂的长度）
        return -3, minRes
    elif 3<minIndex <8: # 左手动作，右手不能动
        print(arr_norm[-1,3,0])
        if 0<arr_norm[-1,3,0]<0.5:
            return -3, minRes
    elif minIndex == 4: # 左上
        if arr_norm[-1,1,0]>-1 or arr_norm[-1,1,1]>-0.2:
            return -3, minRes
    elif minIndex == 5: # 右上
        if arr_norm[-1,1,0]<0.2 or arr_norm[-1,1,1]>-0.2:
            return -3, minRes
    elif minIndex == 6: # 左下
        if arr_norm[-1,1,0]>-1 or arr_norm[-1,1,1]<0.5:
            return -3, minRes
    elif minIndex == 7: # 左下
        if arr_norm[-1,1,0]<0.2 or arr_norm[-1,1,1]<0.5:
            return -3, minRes
    # 放大约束
    elif minIndex == 4:
        print(arr_norm[0,3,0],arr_norm[0,1,0],arr_norm[-1,3,0],arr_norm[-1,1,0])
#         if (0<arr_norm[0,3,0]<0.5 and -0.5<arr_norm[0,1,0]<0  # 起始位置
#             and  arr_norm[-1,3,0]>0.8 and arr_norm[-1,1,0]<0.8): # 起始位置
        return 4, minRes
#         else:
#             return -3, minRes
    # 缩小约束
    elif minIndex == 5:
        print(arr_norm[0,3,0],arr_norm[0,1,0],arr_norm[-1,3,0],arr_norm[-1,1,0])
#         if (0<arr_norm[-1,3,0]<0.5 and -0.5<arr_norm[-1,1,0]<0 # 起始位置
#             and  arr_norm[0,3,0]>0.8 and arr_norm[0,1,0]<0.8): # 起始位置
        return 5, minRes
#         else:
#             return -3, minRes
    elif minIndex >= 6 and arr_norm[-1, 3, 0]>0.5:  # 左手手势，右手保持贴近身体不动
        return -3, minRes
    
    # elif minIndex == 0 and ar_coords[-1,3,1] < 0.5 :
        # return -3,minRes     
    # elif minIndex>=2 and minRes > 80:
        # return -3,minRes
    # if minRes > 50 and minIndex: #上下50 可以；左右不可以。
        # return -3,minRes
    return minIndex, minRes
    # 判断是否满足要求
    dxy = arr_coords[-1,3,:2] - arr_coords[-1,2,:2] #最后一帧 右手部减去肘部的x,y 
    dxyl = arr_coords[-1,1,:2] - arr_coords[-1,0,:2] #最后一帧 左手部减去肘部的x,y 
    #两个手的距离
    dhands = distance(arr_coords[-1,1,:2],arr_coords[-1,3,:2])
    #一只胳膊的距离=大臂+小臂 ~ 小臂*2
    darm = distance(arr_coords[-1,2,:2],arr_coords[-1,3,:2])
    # 0-6:右手 上下左右，放大缩小，左手左移
    if ((minIndex == 0) and (dxy[1] > 0.1+XY_THRESHOLD)) or  \
    ((minIndex == 1) and (dxy[1] < -XY_THRESHOLD)) or         \
    ((minIndex == 2) and (dxy[0] < -XY_THRESHOLD)) or         \
    ((minIndex == 3) and (dxy[0] > XY_THRESHOLD)) or \
    ((minIndex == 4) and (dhands > darm*2)) or         \
    ((minIndex == 5) and (dhands < 0.1)) or         \
    ((minIndex == 6) and (dxyl[0] < -XY_THRESHOLD)):
        return minIndex,minRes
    else:
        return -1,minRes
    

def processGesture(minIndex:int,lastIndex:int,minRes):
    """
    结果执行：根据dtw计算处理的结果，采取相应的操作。
    """
#     assert(minIndex>=-3)
#     assert(minIndex<=6)
#     print(f'show ## minIndex: {minIndex}, minRes: {minRes}')
    if minIndex == -3:
#         print("###################################--匹配过大--手--势--###############################################")
#         print(minRes)
        pass
    elif minIndex == -2:
#         print("###################################--静--止--手--势--###############################################")
        pass
    elif minIndex == -1:
#          print("********************************--无--手--势--*****************************************")
        pass
    elif minIndex==lastIndex :
#         print("+++++++++++++++++++++++++++++++--相--同--手--势--++++++++++++++++++++++++++++++++++++++")
        pass
    else:
        # plotgesture(arr_coords)
#         print("res:",minRes)

#         key_test(list_key_office[minIndex],time_sleep=0)
        show_results(minIndex)

        # key_input(list_key_office[minIndex],time_sleep=0)
        time_end = time.time()
        time_dura = time_end-time_start
#         print(time_dura)
        # list_time.append(time_dura)
#         with open('time.csv','a') as f:
#             f.writelines(str(time_dura)+"\n")
def show_results(minIndex, time_sleep=0):
    if minIndex == 0:
        print('^上↑'*15 )
    elif minIndex == 1:
        print('v下↓'*15 )
    elif minIndex == 2:
        print('<左←'*15 )
    elif minIndex == 3:
        print('>右→'*15 )
    elif minIndex == 4:
        print('↖左上'*10 )
    elif minIndex == 5:
        print('↗右上'*10 )
    elif minIndex == 6:
        print('↙左下'*10 )
    elif minIndex == 7:
        print('↘右下'*10 )
    elif minIndex == 8:
        print('>放大<'*15 )
    elif minIndex == 9:
        print('<缩小>'*15 )
        
def getData(flag_src)->np.array:
    if FLAG_SRC == 1:
        _,arr_cords = get_k2_http(url)
        return arr_cords[-LENGTH_TEMP:] #根据模板的长度，截取相同长度的数据
        # print(arr_coords[-1])
        # print("shape:",arr_coords.shape)
        # plotgesture(arr_coords)
    elif FLAG_SRC == 2:
        _,arr_cords, arr_norm = get_pose()
        if arr_cords is not None:
            return arr_cords[-LENGTH_TEMP:], arr_norm[-LENGTH_TEMP:] #根据模板的长度，截取相同长度的数据
        else:
            return None, None
    else:
        return get_k2_data(arr_coords_saved,i,LENGTH_TEMP), None
        

# 获取模板数据
LENGTH_TEMP = 30
arr_temp,arr_temp_norm = get_templates("gesture_template/0907_01")
# for a in arr_temp_norm:
    # plotgesture(a)
    
FLAG_SRC = 2 # 1 for kinect and 0 for saved data
# 读取保存的k2坐标
if FLAG_SRC == 1:
    url = "http://10.23.155.43:8080/track"
elif FLAG_SRC == 2:
#     _,arr_coords_saved = get_pose()
    pose_estimation()
else:
    _,arr_coords_saved = get_coordinates("data/mppose_1694069093.csv")

## 超参数
# 权重参数调控值
b = 0.1
XY_THRESHOLD = 0.4
# 定义手势判定最小移动距离
GESTURE_MIN_THRESHOLD = 0.1
print("----------------start------------------")
i = 0
list_move = []
while True:
    arr_coords,_ = getData(FLAG_SRC)
    if arr_coords is not None:
        break
    print("确保上半身视频中可见！")
    time.sleep(1)
    
#获取鼠标转换基础数据
# dip = getdim(arr_coords[-1])

# arr_coords,_ = getData(FLAG_SRC)
arr_coords_last = arr_coords.copy()
lastIndex = -1

list_time = []
time_dura = 0
while (True):
    arr_coords, arr_norm = getData(FLAG_SRC)
    if arr_coords is None:
        continue
    time_start = time.time()# 获取数据时刻    
    if arr_coords.shape[0] < LENGTH_TEMP:
        print("arr_coords shorter than the length!")
        continue
    # print('start_cal')
    # mouse_move_coords(arr_coords[-1],*dip)
    minIndex,minRes = cal_dtw(arr_temp_norm,arr_coords,arr_coords_last, arr_norm)
    # print('end_cal')
    # 结果处理
    processGesture(minIndex,lastIndex,minRes)

    lastIndex = minIndex
    arr_coords_last = arr_coords
    i+=1
    time.sleep(1)
