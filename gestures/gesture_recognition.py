#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import _thread
import time
import pyautogui

import cv2 as cv
import numpy
import numpy as np
import mediapipe as mp
import easyocr


from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

GESTURE_MODE = {
    0 : 'CLEAR', #  STOP
    1 : 'OCR', # FIST
    8 : 'MOUSE_UP', #  UP ************************互换 ONE**********************
    2 : 'WRITE', #  OK
    4 : 'MOUSE_DOWN', #  DOWN
    5 : 'POLYGON', # FOUR
    6 : 'MOUSE_LEFT_CLICK', # LEFT
    7 : 'MOUSE_RIGHT_CLICK', # RIGHT
    3 : 'MOUSE_MOVE', # ONE **********************互换 UP***********************
    9 : 'CIRCLE', # YEAH

    10 : 'VIEW'
}

WIDTH = 640
HEIGHT = 480





class GestureRecognition:
    def __init__(self, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                 history_length=16):
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.history_length = history_length

        # 读取模型
        self.hands, self.keypoint_classifier, self.keypoint_classifier_labels, \
        self.point_history_classifier, self.point_history_classifier_labels = self.load_model()

        # 保存板书的点集
        self.point_history = []

        # 手势状态初始
        self.gesture_mode = 'VIEW'

        # 记录当前有几帧是同样的手势
        self.gesture_counter = 0
        self.gesture_id = 0

        # 保存基本图形绘制的
        self.polygons = []
        # self.rectangles = []
        self.circles = []

        # ocr
        self.reader = easyocr.Reader(['ch_sim','en'], gpu=False) # this needs to run only once to load the model into memory
        self.character = []
        self.ocr_flag = 0

        # 保存历史手势
        self.history_gesture_mode = deque(maxlen=history_length)

    def character_recognise(self, shape):
        disImg = np.zeros(shape, dtype=numpy.uint8)
        for j in range(1, len(self.point_history)):
            if self.point_history[j - 1][0] != 0 and self.point_history[j - 1][1] != 0 and self.point_history[j][0] != 0:
                cv.line(disImg, (self.point_history[j - 1][0], self.point_history[j - 1][1]),
                        (self.point_history[j][0], self.point_history[j][1]), (255, 255, 255), 2, 6, 0)
        disImg = 255 - disImg
        self.character = self.reader.readtext(disImg)
        _thread.exit()


    def load_model(self):
        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        return hands, keypoint_classifier, keypoint_classifier_labels, \
               point_history_classifier, point_history_classifier_labels

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
                # 由实际像素转换为相对手腕关键点像素坐标并将坐标归一化
                pre_processed_point_history_list = self._pre_process_point_history(
                    debug_image, self.point_history)

                # Write to the dataset file (mode==0,pass)
                self._logging_csv(number, mode, pre_processed_landmark_list,
                                  pre_processed_point_history_list)


                # 系统模式判断
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == self.gesture_id and self.gesture_counter >= 25: # 如果连续x帧都为当前手势，则系统模式切换
                    self.gesture_mode = GESTURE_MODE[hand_sign_id]
                    self.gesture_counter = 0
                elif hand_sign_id == self.gesture_id : # 如果当前帧手势与上一帧手势相同，但还没有连续x帧都为当前手势，则计数器加一
                    self.gesture_counter += 1
                elif hand_sign_id != self.gesture_id: # 如果当前帧手势与上一帧手势不同，则记录当前帧手势，计数器清零，模式切换为默认模式（观察）
                    self.gesture_counter = 0
                    self.gesture_id = hand_sign_id
                    self.gesture_mode = 'VIEW'


                # 各个模式切换控制
                if self.gesture_mode == 'VIEW' or self.gesture_mode == '':
                    self.point_history.append([0,0])
                elif self.gesture_mode == 'CLEAR':
                    self.point_history.clear()
                elif self.gesture_mode == 'OCR':
                    # if self.point_history:
                    #     _thread.start_new_thread(self.character_recognise, (debug_image.shape, ))
                    #     print(self.character)
                    #     self.point_history.clear()
                    self.point_history.clear()
                    self.ocr_flag = 1
                elif self.gesture_mode == 'MOUSE_UP':
                    pyautogui.scroll(1)
                elif self.gesture_mode == 'WRITE':
                    self.point_history.append(landmark_list[8])
                elif self.gesture_mode == 'MOUSE_DOWN':
                    pyautogui.scroll(-1)
                elif self.gesture_mode == 'POLYGON':
                    basic_graph = []
                    for point in self.point_history:
                        if point != [0, 0]:
                            basic_graph.append(point)
                    if basic_graph:
                        mat_array = np.mat(basic_graph)
                        approx = cv.approxPolyDP(mat_array, 20, True)
                        # x, y, w, h = cv.boundingRect(approx)
                        # self.rectangles.append([x, y, w, h])
                        self.polygons.append(approx)
                        self.point_history.clear()
                elif self.gesture_mode == 'MOUSE_LEFT_CLICK':
                    pyautogui.click()
                elif self.gesture_mode == 'MOUSE_RIGHT_CLICK':
                    pyautogui.rightClick()
                elif self.gesture_mode == 'MOUSE_MOVE':
                    screenWidth, screenHeight = pyautogui.size()
                    ratioWidth = screenWidth/WIDTH
                    ratioHeight = screenHeight/HEIGHT
                    [tempx, tempy] = landmark_list[8]
                    pyautogui.moveTo(tempx * ratioHeight, tempy * ratioWidth)
                elif self.gesture_mode == 'CIRCLE' and self.history_gesture_mode and self.history_gesture_mode[-1] != 'CIRCLEh' :
                    basic_graph = []
                    for point in self.point_history:
                        if point != [0, 0]:
                            basic_graph.append(point)
                    if basic_graph:
                        mat_array = np.mat(basic_graph)
                        approx = cv.approxPolyDP(mat_array, 20, True)
                        (x, y), radius = cv.minEnclosingCircle(approx)
                        self.circles.append([x,y,radius])
                        self.point_history.clear()

                # debug_image = cv.imread("img.png")

                # 可视化
                debug_image = self._draw_bounding_rect(USE_BRECT, debug_image, brect)
                debug_image = self._draw_landmarks(debug_image, landmark_list)
                debug_image = self._draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    GESTURE_MODE[hand_sign_id],
                    self.gesture_mode
                )

                # # Saving gesture
                # self.gesture_id = hand_sign_id
                if self.gesture_mode != 'VIEW':
                    self.history_gesture_mode.append(self.gesture_mode)
        else:
            self.point_history.append([0, 0])
            hand_sign_id = -1
            self.gesture_id = 0

        show_image = self.draw_point_history(debug_image, self.point_history)
        

        return debug_image,show_image, hand_sign_id
        # return debug_image, [k for k, v in GESTURE_MODE.items() if v == self.gesture_mode][0]


    def draw_point_history(self, image, point_history):
        black_background = np.zeros((480, 640), dtype=np.uint8)
        for j in range(1, len(point_history)):
            if point_history[j - 1][0] != 0 and point_history[j - 1][1] != 0 and point_history[j][0] != 0:
                cv.line(black_background, (point_history[j - 1][0], point_history[j - 1][1]),
                     (point_history[j][0], point_history[j][1]), (255, 255, 255), 2, 6, 0)

        for polygon in self.polygons:
            cv.polylines(image, [polygon], True, (255, 0, 0), 2)

        # for rect in self.rectangles:
        #     cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,0,0), 2)

        for circle in self.circles:
            cv.circle(image, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 2)


        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        if self.ocr_flag == 1:
            cv.putText(image, "AI", (200, 300), cv.FONT_HERSHEY_SIMPLEX, 10.0, (255, 255, 0), 2, cv.LINE_AA)


        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
        return image

    def _logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            print("WRITE")
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

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

    def _pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def _draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Wrist 1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # Wrist 2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # Index finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # Middle finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: point first
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # Ring finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # Little finger: base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: point first
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def _draw_info_text(self, image, brect, handedness, hand_sign_text,
                        finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)

        return image

    def _draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image
