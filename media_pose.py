import os
import time

import csv
import cv2
import mediapipe as mp

kp_path = f'data/mppose_{int(time.time())}.csv'

# 初始化MediaPipe的人体姿态组件
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# 初始化视频输出
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(f'data/videos/hand_gesture_{int(time.time())}.avi')

# 创建CSV文件并写入标题行
with open(kp_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        "HandLeftX", "HandLeftY", "HandLeftZ",
        "HandRightX", "HandRightY", "HandRightZ",
        "ShoulderLeftX", "ShoulderLeftY", "ShoulderLeftZ",
        "ShoulderRightX", "ShoulderRightY", "ShoulderRightZ",
        "hipLeftX", "hipLeftY", "hipLeftZ",
        "hipRightX", "hipRightY", "hipRightZ"
    ])

# 创建一个Pose对象，用于运行人体姿态估计
# 打开摄像头
cap = cv2.VideoCapture(0)
frame_count = -1
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        time_start = time.time()
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
        # 保存特定关节点的二维坐标和均值到CSV
        if results.pose_landmarks:
            row = []
            specified_landmarks = [16, 15, 12, 11, 24, 23]
            if all(results.pose_landmarks.landmark[id].visibility > 0.5 for id in specified_landmarks):
                # out.write(image)
                for id in specified_landmarks:
                    landmark = results.pose_landmarks.landmark[id]
                    x, y, z = landmark.x, landmark.y, landmark.z
                    row.extend([x, y, z])
                with open(kp_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(row)
        time_used = time.time()-time_start
        fps = 1//time_used
        print(fps)
        # 输出视频
        cv2.putText(image, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # 显示图像
        cv2.imshow('MediaPipe Pose', image)
        # 按下'q'键退出循环
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
# out.release()
cv2.destroyAllWindows()
