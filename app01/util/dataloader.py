import os
import time
import glob

import pandas as pd
import numpy as np
import requests
import cv2
import mediapipe as mp

from threading import Thread
from models.modules import coord_norm 

#---------------------------------------------------------------------------
# 数据获取
#---------------------------------------------------------------------------
# midiapipe 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class Holder():
    list_row = []
    
holder = Holder()
def get_pose():
    if len(holder.list_row) > 30:
        list_row_yield = holder.list_row.copy()
        arr_co = np.array(list_row_yield).reshape(-1,6,3)
        # 左肘、左手、右肘、右手 相对躯干的距离
        arr_co_norm  = np.array(list(map(coord_norm,arr_co)))
        arr_co_diff  = arr_co_norm[1:] - arr_co_norm[:-1]    
        return arr_co, arr_co_diff, arr_co_norm
    return None,None,None

def pose_estimation():
    print('pose estimation')
    def run():
#         assert os.path.exists("data/videos/test1.mp4"), "video does not exist"
#         cap = cv2.VideoCapture("data/videos/test2.mp4")
        cap = cv2.VideoCapture(0)
        frame_count = -1
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                # mirror image
                # print(image.shape)
                image = image[:,::-1, :].copy()
                frame_count += 1
                # 将BGR图像转换为RGB图像
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 进行人体姿态估计
                results = pose.process(image_rgb)

                # 在图像上绘制人体关节点
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 输出视频
                cv2.putText(image, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # 保存特定关节点的二维坐标和均值到CSV
                if results.pose_landmarks:
                    row = []
                    direct_landmarks = [14, 16, 13, 15]
                    mean_landmarks_1 = [11, 12]
                    mean_landmarks_2 = [11, 12, 23, 24]
                    specified_landmarks = [14, 16, 13, 15, 12, 11, 23, 24]
                    if all(results.pose_landmarks.landmark[id].visibility > 0.5 for id in specified_landmarks):
                        # out.write(image)
                        for id in direct_landmarks:
                            landmark = results.pose_landmarks.landmark[id]
                            x, y,z = landmark.x ,landmark.y ,landmark.z
                            row.extend([x, y,z])

                        mean_x_1 = mean_y_1=mean_z_1 = mean_x_2 = mean_y_2 =mean_z_2=0

                        for id in mean_landmarks_1:
                            landmark = results.pose_landmarks.landmark[id]
                            mean_x_1 += landmark.x 
                            mean_y_1 += landmark.y 
                            mean_z_1+=landmark.z

                        mean_x_1 /= len(mean_landmarks_1)
                        mean_y_1 /= len(mean_landmarks_1)
                        mean_z_1/=len(mean_landmarks_1)

                        for id in mean_landmarks_2:
                            landmark = results.pose_landmarks.landmark[id]
                            mean_x_2 += landmark.x 
                            mean_y_2 += landmark.y 
                            mean_z_2+=landmark.z
                        mean_x_2 /= len(mean_landmarks_2)
                        mean_y_2 /= len(mean_landmarks_2)
                        mean_z_2/=len(mean_landmarks_2)

                        row.extend([mean_x_1, mean_y_1,mean_z_1, mean_x_2, mean_y_2,mean_z_2])
                        holder.list_row.append(row)

                if len(holder.list_row)>100:
                    holder.list_row = holder.list_row[-30:].copy()
                # 显示图像
                cv2.imshow('MediaPipe Pose', image)

                # 按下'q'键退出循环
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    # out.release()
                    cv2.destroyAllWindows()
                    break
    Thread(target=run,).start()

# 从kinect实时读取
def get_k2_http(url):
    """
    获取实时数据
    获取k2数据
    """
    r = requests.get(url)
    rs = r.text.split("\n")
    list_data = []
    for r in rs:
        if len(r) == 0:
            continue
        list_data.append([eval(n) for n in r.split(",")])
    arr_co = np.array(list_data,dtype=np.float32).reshape(-1,6,3)

    arr_co_norm  = np.array(list(map(coord_norm,arr_co)))

    return  arr_co,arr_co_norm 

def test_httpapi():
    url = "http://10.24.33.54:8080/track"
    print(get_k2_http(url)[1].shape)

# 从单个csv文件中读取数据
def get_coordinates(file_path):
    """
    从csv文件中 读取坐标文件
    """
    pd_1 = pd.read_csv(open(file_path))
    arr_coords = pd_1.values.reshape(-1,6,3) #
    arr_coords_norm = np.array(list(map(norm_pose,arr_coords)))
    arr_coords_norm = arr_coords_norm[1:] - arr_coords_norm[:-1]
    return arr_coords, arr_coords_norm


def norm_pose(arr_poses):
    """给定左右手、肩、胯的三维坐标，对手的坐标进行规范化
    手以肩为原点，以肩到胯的长度为单位长度；
    输入： list_poses:
                  O
    (16)=---(12)+---+(11)---=(15)
                |   |
                |   |
            (24)+---+(23)

    入上图所示，分别为右/左的手、肩、胯的(x,y,z)坐标，注意图像镜像过了
    """
    # print(arr_poses.shape)
    arr_hands_norm = (arr_poses[:2,:]-arr_poses[2:4,:])/\
        np.expand_dims(np.sqrt(((arr_poses[4:, :]-arr_poses[2:4,:])**2).sum(axis=1)),1).repeat(3,axis=1)
    return arr_hands_norm


# 从多个csv文件中读取模板数据
def get_templates(path,suffix = ".csv"):
    """
    返回模板数据，给定模板数据所在目录，自动读取其中的所有csv文件，将其取出，并
    返回归一化后的结果和之前的结果
    """
    list_ts = []
    list_ots = []
    for f in glob.glob(os.path.join(path,"*"+suffix)):
        # print(f)
        arr_co, arr_co_norm = get_coordinates(f)
        list_ts.append(arr_co)
        list_ots.append(arr_co_norm)

    return np.array(list_ts), np.array(list_ots) 

# 从txt文件中读取（k1模板数据）
def getgesture(filename):
    """
    读取原模板数据 
    """
    print("读入模板数据：",filename)
    xs = []
    ys = []
    zs = []
    with open(filename) as f:
        for line in f.readlines():
    #         print(line)
            if line.startswith("---"):
                continue
            res = line.split(",")
            if len(res) == 3:
                x,y,z = res
                xs.append(min([eval(x),2]))
                ys.append(min([eval(y),2]))
                zs.append(min([eval(z),2]))
            else:
                print(line)
    return [xs,ys,zs]

def getTempGestures(path):
    """
   读取模板文件目录下的所有txt文件，并返回形如7*l*g*3的numpy数组；
   其中，7为手势类型，目前为7种;l为每个模板数据的长度，目前为20个；
   g为数据点的个数，目前取4个手部点；3为3维坐标。 
    """
    list_gestures = []
    for fname in glob.glob(os.path.join(path,"*.txt")):
        list_gestures.append(getgesture(fname))
    return np.array(list_gestures).transpose(0,2,1).reshape(7,-1,4,3)

def test1():
    abs_path = r"E:\Music\interaction_project"
    file_path = "data_k2/data_sun.csv"
    a1,a2 = get_coordinates(os.path.join(abs_path,file_path))
    print(a1.shape,a2.shape)

def test_get_templates():
    abs_path = r"E:\Music\interaction_project"
    file_path = "gesture_template/30"
    a1,a2 = get_templates(os.path.join(abs_path,file_path))
    print(a1.shape,a2.shape)
    
if __name__ == "__main__":
    test_get_templates()