import os
import tempfile

import cv2
import mediapipe as mp
import math
import time
import numpy as np

# 将视频二进制数据保存为临时文件
# with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
#     temp_file.write(bytes_data)
#     temp_file_path = temp_file.name
# # 使用临时文件路径作为视频源进行处理
# video_stream = cv2.VideoCapture(temp_file_path)


# ======================这个是原始调摄像头的上下左右滑动=============================


# # 初始化MediaPipe Hand模型
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # 打开摄像头
# cap = cv2.VideoCapture(0)
# # 初始化手指位置和时间
# prev_center_x = None
# prev_center_y = None
# prev_time = time.time()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # 将图像转换为RGB格式
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # 使用MediaPipe检测手部关键点
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             keypoints = [
#                 hand_landmarks.landmark[0],
#                 hand_landmarks.landmark[1],
#                 hand_landmarks.landmark[2],
#                 hand_landmarks.landmark[5],
#                 hand_landmarks.landmark[9],
#                 hand_landmarks.landmark[13],
#                 hand_landmarks.landmark[17]
#             ]
#             # 提取关键点的坐标
#             keypoint_coords = [(int(kp.x * frame.shape[1]), int(kp.y * frame.shape[0])) for kp in keypoints]
#
#             # 计算凸包
#             try:
#                 hull = cv2.convexHull(np.array(keypoint_coords))
#                 # 计算凸包的中心点
#                 M = cv2.moments(hull)
#                 center_x = int(M["m10"] / M["m00"])
#                 center_y = int(M["m01"] / M["m00"])
#                 # 显示中心点
#                 cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
#                 # 计算时间间隔
#                 current_time = time.time()
#                 time_elapsed = current_time - prev_time
#
#                 if prev_center_x is not None and time_elapsed > 0:
#                     x_diff = abs(center_x - prev_center_x) / time_elapsed
#                     y_diff = abs(center_y - prev_center_y) / time_elapsed
#
#                     # 判断手势
#                     if x_diff > y_diff:
#                         if center_x > prev_center_x:
#                             gesture_result="向左滑动"
#                             # print("向右滑动")
#                         else:
#                             gesture_result="向右滑动"
#                             # print("向左滑动")
#                     elif y_diff > x_diff:
#                         if center_y > prev_center_y:
#                             gesture_result="向下滑动"
#                             # print("向下滑动")
#                         else:
#                             gesture_result="向上滑动"
#                             # print("向上滑动")
#
#                 prev_center_x = center_x
#                 prev_center_y = center_y
#                 prev_time = current_time
#                 print(gesture_result)
#
#             except Exception as e:
#                 print("Error calculating convex hull:", str(e))
#
#     # 显示图像
#     cv2.imshow('Hand Gesture Recognition', frame)
#
#     # 按下'q'键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放摄像头并关闭窗口
# cap.release()
# cv2.destroyAllWindows()

# =====================================

import cv2
import mediapipe as mp
import math
import time
import numpy as np


def analyze_hand_gesture(bytes_data):
    # 初始化MediaPipe Hand模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    print(11111)

    # 将视频二进制数据保存为临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(bytes_data)
        print(temp_file.name)
        temp_file_path = temp_file.name
    # 使用临时文件路径作为视频源进行处理
    # time.sleep(3)
    video_stream = cv2.VideoCapture(temp_file_path)
    # print(222)
    
    # video_stream = cv2.VideoCapture()
    # video_stream.open(''.join(map(chr, bytes_data)))

    # os.remove(temp_file_path)

    # # 将视频流解码
    # cap = cv2.VideoCapture(0)
    # # cap.open(0)

    # 初始化手指位置和时间
    prev_center_x = None
    prev_center_y = None
    prev_time = time.time()
    # print(333)

    while True:
        # print(555)
        ret, frame = video_stream.read()
        # print(444)
        if not ret:
            break
        # 使用MediaPipe检测手部关键点
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(2)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = [
                    hand_landmarks.landmark[0],
                    hand_landmarks.landmark[1],
                    hand_landmarks.landmark[2],
                    hand_landmarks.landmark[5],
                    hand_landmarks.landmark[9],
                    hand_landmarks.landmark[13],
                    hand_landmarks.landmark[17]
                ]
                print(3)
                # 提取关键点的坐标
                keypoint_coords = [(int(kp.x * frame.shape[1]), int(kp.y * frame.shape[0])) for kp in keypoints]
                print(4)
                gesture_result = ""

                # 计算凸包
                try:
                    hull = cv2.convexHull(np.array(keypoint_coords))
                    # 计算凸包的中心点
                    M = cv2.moments(hull)
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    # 计算时间间隔
                    current_time = time.time()
                    time_elapsed = current_time - prev_time

                    if prev_center_x is not None and time_elapsed > 0:
                        x_diff = abs(center_x - prev_center_x) / time_elapsed
                        y_diff = abs(center_y - prev_center_y) / time_elapsed

                        # 判断手势
                        if x_diff > y_diff:
                            if center_x > prev_center_x:
                                gesture_result = "向左滑动"

                            else:
                                gesture_result = "向右滑动"

                        elif y_diff > x_diff:
                            if center_y > prev_center_y:
                                gesture_result = "向下滑动"

                            else:
                                gesture_result = "向上滑动"

                    prev_center_x = center_x
                    prev_center_y = center_y
                    prev_time = current_time
                    print(5)

                    return gesture_result

                except Exception as e:
                    print("Error calculating convex hull:", str(e))

        # 显示图像
        # cv2.imshow('Hand Gesture Recognition', frame)

        # # 按下'q'键退出循环
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放摄像头并关闭窗口
    print(7)
    video_stream.release()
    cv2.destroyAllWindows()
    print(6)

    return '2222'

