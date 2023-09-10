import cv2 as cv

import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import re
def get_direct_paths(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)]
def find_and_convert(s):
    # 使用正则表达式查找'0'后的所有数字字符
    match = re.search('0(\d+)', s)
    
    # 如果找到匹配项，返回转换后的数字
    if match:
        return int(match.group(1))
    # 如果没有匹配项，返回None
    return None

class GestureRecognition:
    def __init__(self, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                 history_length=16):
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.history_length = history_length

        # 保存板书的点集
        self.point_history = []

        # 手势状态初始
        self.gesture_mode = 'VIEW'

        # 记录当前有几帧是同样的手势
        self.gesture_counter = 0
        self.gesture_id = 0

        # 读取mediapipe
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def recognize(self, image, number=13, mode=0):

        # bounding_rect
        USE_BRECT = True

        image = cv.flip(image, 1)  # Mirror display
        # debug_image=cv.imread("image.png")
        debug_image = copy.deepcopy(image)
        hand_sign_id = 0

        ############Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        #####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 计算手部矩形框
                brect = self._calc_bounding_rect(debug_image, hand_landmarks)
                # 关键点坐标计算（由比例转为实际像素）
                landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)

                # 由实际像素转换为相对手腕关键点像素坐标并将坐标归一化
                pre_processed_landmark_list = self._pre_process_landmark(
                    landmark_list)

                # Write to the dataset file (mode==0,pass)
                self._logging_csv(number, mode, pre_processed_landmark_list)

        else:
            pass

        show_image = image
        

        return debug_image,show_image, hand_sign_id
        # return debug_image, [k for k, v in GESTURE_MODE.items() if v == self.gesture_mode][0]


    def _logging_csv(self, number, mode, landmark_list):
        # print("WRITE")
        csv_path = 'keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    def _calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def _calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def _pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


gesture_detector = GestureRecognition()
# Specify the directory containing your images

directory = r'C:\Users\25352\Desktop\hand-gester\handpose_x_gesture_v1'
direct_paths = get_direct_paths(directory)
for tem in direct_paths:
    image_directory =tem

        # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]  # you can add more extensions if needed
    # print(image_files)
    print(tem)
    mode = 0
    number = find_and_convert(image_directory)
    print(number)

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv.imread(image_path)
        _,_,_ = gesture_detector.recognize(image, number, mode)
    

