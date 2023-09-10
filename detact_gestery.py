import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from split_train_test import *
from utils import *
import time
import pandas as pd

from util.dataloader import get_templates, norm_pose
from config import DynamicGesture as DG
from models.dtw import dtw
from models.modules import coord_norm,distance,vDistance,plotgesture,vDistance2
from gester_encoder import *
import math
from util.wincontrol import mouse_move


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, num_nodes=21):
        super(TransformerModel, self).__init__()

        # Generate position encoding
        position = torch.arange(num_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.position_enc = torch.zeros(num_nodes, d_model)
        self.position_enc[:, 0::2] = torch.sin(position * div_term)
        self.position_enc[:, 1::2] = torch.cos(position * div_term)

        # Add learnable group embedding
        self.embedding = nn.Linear(2, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 21, 2)  # x shape becomes [batch_size, 21, 2]

        # Embedding the input
        x = self.embedding(x)  # x shape becomes [batch_size, 21, d_model]

        # Add position encoding
        x += self.position_enc

        x = x.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model] as expected by Transformer
        x = self.encoder(x)
        x = self.classifier(x[0])  # Use the first token for classification
        return x
    def load_model(self,path):
        return torch.load(path, map_location=torch.device('cpu'))
# from queue import Queue

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

name = ['one', 'five', 'fist', 'ok', 'seven', 'two', 'three', 'four', 'six', 'I love you','eight','thumb up','nine','pink']


def get_pose_coords(results):
    specified_landmarks = [16, 15, 12, 11, 24, 23]
    row = None
    if all(results.pose_landmarks.landmark[id].visibility > 0.5 for id in specified_landmarks):
        row = []
        for id in specified_landmarks:
            landmark = results.pose_landmarks.landmark[id]
            x, y, z = landmark.x, landmark.y, landmark.z
            row.extend([x, y, z])
    return row


def cal_dtw(dict_coord_hands, arr_temp):
    arr_poses = dict_coord_hands['arr_poses'].copy()
    arr_coord_hands = dict_coord_hands['arr_pose_norm'].copy()
    ar_coords = dict_coord_hands['arr_pose_diff'].copy()
    # 判断是否移动了: 左手或者右手移动半个身位
    res1 = np.sqrt(((arr_coord_hands[-1,:,:] - arr_coord_hands[0,:,:])**2).sum(axis=1))
    res2 = np.sqrt(((arr_poses[-1,:,:] - arr_poses[0,:,:])**2).sum(axis=1))
    GESTURE_MIN_THRESHOLD = 0.7
    COORDS_MIN_THRESHOLD = 0.0
    norm_flag = sum([1 if a >= GESTURE_MIN_THRESHOLD else 0 for a in res1])
    distance_flag = sum([1 if a >= COORDS_MIN_THRESHOLD else 0 for a in res2])
    if (distance_flag & norm_flag) == 0:
        return None
    # print(f'move flag: {res1}')
    bm = np.array([1]*10)
    list_dtw = []
    for t in range(arr_temp.shape[0]):
        # print(ar_coords.shape,arr_temp.shape)
        res,_,_,_ = dtw(ar_coords[:,:,:2],arr_temp[t][:,:,:2],vDistance2,w=4)
        list_dtw.append(res)
    arr_dtw = np.array(list_dtw)
    arr_res = bm[:arr_temp.shape[0]]*arr_dtw
    # arr_res = arr_dtw
    minIndex = np.argmin(arr_res)
    minRes = arr_res[minIndex]
    print(f"minIndex:#{minIndex} minRes={minRes}")
    if minRes > 1.5:
        if minIndex > 7:
            print('xxxxxxForxxxxxMinResxxxxxxxxxxxxxx')
        return None
    # =========================================================================
    # 对动作进行约束：
    # 分别对左手右手的横纵坐标进行限制，看哪些手势需要满足这些条件
    #
    # 注意这里的arr_coord_norm是左右手规范化的坐标
    # 其原点为各自肩部位置， 单位长度为肩部到胯部的距离
    # ========================================================================
    return minIndex
    print(f"#{minIndex} --{arr_coord_hands[-1]}")
    # *右手左右小范围移动，(-0.3, 0.3)*
    if -0.3 < arr_coord_hands[-1, 1, 0] < 0.3:  # 右手的横坐标不超过0.5;
        if gesture_id not in [0, 1, 4, 5, 6, 7, 9]:  # 限定右手横坐标
            return None
    # *右手上下小范围移动，(-0.2, 0.2)*
    # if -0.2 < arr_coord_hands[-1, 1, 1] < 0.2:  # 右手的横坐标不超过0.5;
    #     if gesture_id not in [2, 3, 4, 5, 6, 7, ]:  # 限定右手横坐标
    #         return None
    # *左手左右小范围移动，(-0.3, 0.3)*
    if -0.3 < arr_coord_hands[-1, 0, 0] < 0.3:  # 右手的横坐标不超过0.5;
        if gesture_id not in [0, 1, 2, 3, 9]:  # 限定右手横坐标
            return None
    # *左手上下小范围移动，(-0.2, 0.2)*
    # if -0.2 < arr_coord_hands[-1, 0, 1] < 0.2:  # 右手的横坐标不超过0.5;
    #     if gesture_id not in [0, 1, 2, 3]:  # 限定右手横坐标
    #         return None
    return minIndex


class GestureRecognition:
    def __init__(self, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # 保存板书的点集
        self.point_history = []

        # 记录当前有几帧是同样的手势
        self.gesture_counter = 0
        self.gesture_id = 0

        # 读取mediapipe
        self.mp_hands = mp.solutions.hands
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=4,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier =  TransformerModel(d_model=24, nhead=4, num_layers=3, num_classes=14)
        # 加载整个模型
        self.keypoint_classifier = self.keypoint_classifier.load_model(r'best_modelv5.pth').to('cpu')
        self.keypoint_classifier.position_enc=self.keypoint_classifier.position_enc.to('cpu')
        self.keypoint_classifier.eval()
        self.frame=-1
        self.left_queue = FixedSizeQueue(60)
        self.right_queue = FixedSizeQueue(60)
        self.time_gap=25
        self.non_gester=1
        self.catch_gester=0
        self.previous_land_mark=None
        self.current_land_mark=None
        self.threshold=0.2
        self.result_queue=FixedSizeQueue(2)
        self.result_queue.push([-1,-1])
        self.move_flag=0
        self.catch_flag=0
        # todo: template
        self.template_dynamic_gesture = get_templates(r"gesture_template\0908")[1]
        # self.image_holder = FixedSizeQueue(30)
        self.pose_queue = FixedSizeQueue(20)
        self.mouse_queue = FixedSizeQueue(10)

    def _mouse_move(self, image):
        results = self.pose.process(image)
        if results is None:
            return None
        if results.pose_landmarks is  None:
            return None
        # drawing keypoints
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # results parsing
        frame_pose_coords = get_pose_coords(results)
        if frame_pose_coords is None:
            # print('WARNING: 动态手势未检测到足够多的关键点\n'
            #       '请确保手、肩、胯在视野内！')
            # time.sleep(1)
            return None
        if self.frame % 4 == 0:
            self.mouse_queue.push(frame_pose_coords)
        if self.mouse_queue.full(): 
            if self.frame % 10 != 0:
                return None
            arr_poses = np.array(self.pose_queue.get_all()).reshape(-1,6,3)
            # 最后一帧和前一帧的相对位移，
            delta = arr_poses[-1,:2,:2] - arr_poses[0,:2,:2]
            print('mouse move: {delta}')
            mouse_move(*delta.tolist())

    self.lookup = GestureLookup(r'C:\Users\25352\Desktop\map.csv')

    def _dynamic_gesture(self,image) -> int:
        # pose estimation
        results = self.pose.process(image)
        if results is None:
            return None
        if results.pose_landmarks is  None:
            return None
        # drawing keypoints
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # results parsing
        frame_pose_coords = get_pose_coords(results)
        if frame_pose_coords is None:
            # print('WARNING: 动态手势未检测到足够多的关键点\n'
            #       '请确保手、肩、胯在视野内！')
            # time.sleep(1)
            return None
        if self.frame % 4 == 0:
            self.pose_queue.push(frame_pose_coords)
        # print(f'len:{len(self.pose_queue)}')
        if self.pose_queue.full(): 
            if self.frame % 40 != 0:
                self.move_flag = 1
                return None
            arr_poses = np.array(self.pose_queue.get_all()).reshape(-1,6,3)
            # arr_coord_hands = norm_pose(arr_poses)
            arr_coord_hands = np.array(list(map(norm_pose, arr_poses)))
            arr_coord_diff = arr_poses[1:, :2, :] - arr_poses[:-1, :2, :]
            dict_coord_hands = {
                "arr_poses": arr_poses,
                "arr_pose_norm": arr_coord_hands,
                "arr_pose_diff": arr_coord_diff,
            }
            # calculate the gesture
            gesture_id = cal_dtw(dict_coord_hands, self.template_dynamic_gesture)
            if gesture_id is None:
                return None
            time_used = round(time.time() - time_start,2)
            print(f"动态手势：{gesture_id}, 用时: {time_used}秒")
            self.move_flag=0
            # time.sleep(1)
            return gesture_id
        return None

    def recognize(self, image, number=13, mode=0):
        # bounding_rect
        self.frame=self.frame+1
        result_hand_sign_id = -1
        ############Detection implementation #############################################################
        image.flags.writeable = False
        if not self.move_flag and not self.catch_flag:
            results = self.hands.process(image)
            image.flags.writeable = True
            #####################################################################
            if results.multi_hand_landmarks:
                # Pair each hand with its index
                indexed_hands = list(enumerate(results.multi_hand_landmarks))
                
                # Sort indexed hands based on the average z value (depth) of landmarks
                sorted_indexed_hands = sorted(
                    indexed_hands, 
                    key=lambda idx_hand: sum([landmark.z for landmark in idx_hand[1].landmark]) / len(idx_hand[1].landmark)
                )
                # Extracting sorted indices
                sorted_indices = [idx_hand[0] for idx_hand in sorted_indexed_hands][:2]

                # Get the front two hands
                front_two_hands = []
                for tem in sorted_indices:
                    front_two_hands.append(results.multi_hand_landmarks[tem])
                two_hand_result=[None,None]
                for hand_idx, landmarks in enumerate(front_two_hands):
                    # print("hand_idx",hand_idx)
                   
                    handness = results.multi_handedness[sorted_indices[hand_idx]].classification[0].label  # This will give either 'Right' or 'Left'
                    mp_drawing.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmark_list = calc_landmark_list(image, landmarks)
                        # 由实际像素转换为相对手腕关键点像素坐标并将坐标归一化
                    pre_processed_landmark_list = torch.Tensor(pre_process_landmark(
                            landmark_list))
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list).reshape(14)
                    _, hand_sign_id = torch.max(hand_sign_id.data, 0)
                    
                    if handness=='Left':
                        # print('left')
                        two_hand_result[0]=hand_sign_id
                        # print(two_hand_result[0])
                    else:
                        # print('right')
                        two_hand_result[1]=hand_sign_id
                        # print(two_hand_result[1])
                # print('Left:',name[two_hand_result[0]],'Right',name[two_hand_result[1]])
                self.left_queue.push(two_hand_result[0])
                self.right_queue.push(two_hand_result[1])
                # print(two_hand_result)

                if self.frame%self.time_gap==0:
                    print()
                    two_hand_result[0]=self.left_queue.most_common_in_last_n()[0]
                    if two_hand_result[0] is not None:
                        self.left_queue.clear()
                        
                    two_hand_result[1]=self.right_queue.most_common_in_last_n()[0]
                    if two_hand_result[1] is not None:
                        
                        self.right_queue.clear()
                    self.result_queue.push(two_hand_result)
                    left_hand = name[two_hand_result[0]] if two_hand_result[0] is not None else "Unknown"
                    right_hand = name[two_hand_result[1]] if two_hand_result[1] is not None else "Unknown"
                    left_hand_id=two_hand_result[0] if two_hand_result[0] is not None else None
                    right_hand_id = two_hand_result[1] if two_hand_result[1] is not None else None
                    if two_hand_result[1]==self.non_gester or two_hand_result[0]==self.non_gester:
                        res=list(self.result_queue.queue)
                        self.move_flag=1
                        for tem in res:
                            if self.non_gester not in tem:
                                self.move_flag=0
                                break
                        if self.move_flag:
                            print("进入动态")
                    elif two_hand_result[1]==self.catch_gester:
                        res=list(self.result_queue.queue)
                        self.catch_flag=1
                        for tem in res:
                            if self.catch_gester not in tem:
                                self.catch_flag=0
                                break
                        if self.catch_flag:
                            print("进入追踪")
                    else:
                        result_hand_sign_id=encode(left_hand_id,right_hand_id)
                        print(self.frame,'Left:', left_hand, 'Right:', right_hand)
                        print(result_hand_sign_id)
                        key=self.lookup.get_gesture_by_id(result_hand_sign_id)
                        print("key",key)
                        press_keys_from_string(key)
            else:
                pass
        if self.move_flag:
            result_hand_sign_id = self._dynamic_gesture(image)
            result_hand_sign_id=encode(dynamic=result_hand_sign_id)
            key=self.lookup.get_gesture_by_id(result_hand_sign_id)
            press_keys_from_string(key)
        elif self.catch_flag:
            
            #追踪代码
            self.catch_flag=0
            
        show_image = image.copy()
        return show_image, result_hand_sign_id


if __name__=='__main__':
    gesture_detector = GestureRecognition()
    cap = cv.VideoCapture(0)
    mode = 0
    number = -1
    while True:
        time_start = time.time()
        ret, image = cap.read()
        image = cv.flip(image, 1)  # Mirror display
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        show_image, gesture_id = gesture_detector.recognize(image, number, mode)
        # fps
        time_used = time.time()-time_start
        fps = 1//time_used
        cv.putText(show_image, f'fps: {round(fps, 2)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(show_image, f'Gesture ID: {gesture_id}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('Character writing', show_image)
        if cv.waitKey(10) == ord('q'):  # 点击视频，输入q退出
            break
    cv.destroyAllWindows()
