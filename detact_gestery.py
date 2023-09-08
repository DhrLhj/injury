import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from split_train_test import *
from utils import *
import time

mp_drawing = mp.solutions.drawing_utils
name=['one','five','fist','ok','seven','two','three','four','six','I love you','eight','thumb up','nine','pink']

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
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier =  SimpleModel()
        # 加载整个模型
        self.keypoint_classifier = self.keypoint_classifier.load_model('best_model.pth')
        self.keypoint_classifier.eval()
        self.frame=-1
        self.left_queue = FixedSizeQueue(60)
        self.right_queue = FixedSizeQueue(60)
        self.time_gap=20
        self.non_gester=1
        self.previous_land_mark=None
        self.current_land_mark=None
        self.threshold=0.2
        self.result_queue=FixedSizeQueue(3)
        self.result_queue.push([-1,-1])
        self.move_flag=0

    def recognize(self, image, number=13, mode=0):
        # bounding_rect
        self.frame=self.frame+1
        image = cv.flip(image, 1)  # Mirror display
        hand_sign_id = 0
        ############Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        if not self.move_flag:
            results = self.hands.process(image)
            image.flags.writeable = True
            #####################################################################
            if results.multi_hand_landmarks:
                two_hand_result=[None,None]
                for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                    handness = results.multi_handedness[hand_idx].classification[0].label  # This will give either 'Right' or 'Left'
                    mp_drawing.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmark_list = calc_landmark_list(image, landmarks)
                        # 由实际像素转换为相对手腕关键点像素坐标并将坐标归一化
                    pre_processed_landmark_list = torch.Tensor(pre_process_landmark(
                            landmark_list))
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list).reshape(14)
                    _, hand_sign_id = torch.max(hand_sign_id.data, 0)
                    
                    if handness=='Left':
                        two_hand_result[0]=hand_sign_id
                    else:
                        two_hand_result[1]=hand_sign_id
                # print('Left:',name[two_hand_result[0]],'Right',name[two_hand_result[1]])
                self.left_queue.push(two_hand_result[0])
                self.right_queue.push(two_hand_result[1])

                if self.frame%self.time_gap==0:
                    two_hand_result[0]=self.left_queue.most_common_in_last_n()[0]
                    two_hand_result[1]=self.right_queue.most_common_in_last_n()[0]
                    self.result_queue.push(two_hand_result)
                    left_hand = name[two_hand_result[0]] if two_hand_result[0] is not None else "Unknown"
                    right_hand = name[two_hand_result[1]] if two_hand_result[1] is not None else "Unknown"
                    print(self.frame,'Left:', left_hand, 'Right:', right_hand)
                    if two_hand_result[1]==self.non_gester:
                        print("进入动态")
                        self.move_flag=1

            else:
                pass
        else:
            time.sleep(3)
            self.move_flag=0
        show_image = image
        return show_image, hand_sign_id
       
if __name__=='__main__':
    gesture_detector = GestureRecognition()
    cap = cv.VideoCapture(0)
    mode = 0
    number = -1
    while True:
        ret, image = cap.read()
        show_image, gesture_id = gesture_detector.recognize(image, number, mode)
        cv.imshow('Character writing', show_image)
        if cv.waitKey(1) == ord('q'):  # 点击视频，输入q退出
            break
    cv.destroyAllWindows()