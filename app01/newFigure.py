import os
import tempfile

import mediapipe as mp
import cv2
import numpy as np
import base64
import time

def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    return angle

def get_str_guester(left_up_fingers,right_up_fingers,list_lms):
    str_guester = ''
    
    if len(right_up_fingers)==1 and right_up_fingers[0]==8:
        
        v1 = list_lms[6]-list_lms[7]
        v2 = list_lms[8]-list_lms[7]
        
        angle = get_angle(v1,v2)
        
        if angle<160:
            str_guester = "9"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 0 and right_up_fingers[0] == 8:
        str_guester = "1"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 1 and right_up_fingers[0] == 8 and left_up_fingers[0] == 8:
        str_guester = "1_1"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 2 and right_up_fingers[0] == 8 and left_up_fingers[0] == 8 and left_up_fingers[1] ==12:
        str_guester = "1_2"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 3 and right_up_fingers[0] == 8 and left_up_fingers[0] == 8 and left_up_fingers[1] ==12 and left_up_fingers[0] == 16  :
        str_guester = "1_3"
    

    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 0 and right_up_fingers[0] == 4 and len(left_up_fingers) == 0:
        str_guester = "good"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 1 and right_up_fingers[0] == 4 and left_up_fingers[0] == 4:
        str_guester = "index"

   
    elif len(right_up_fingers) == 1 and right_up_fingers[0] == 16 and len(left_up_fingers) == 0:
        str_guester = "aa"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 2 and right_up_fingers[0] == 16 and left_up_fingers[0] == 16:
        str_guester = "aa_aa"


    elif len(right_up_fingers)==1 and right_up_fingers[0]==20 and len(left_up_fingers) == 0:
        str_guester = "Bad"
    elif len(right_up_fingers) == 1 and len(left_up_fingers) == 1 and right_up_fingers[0] == 20 and left_up_fingers[0] == 20:
        str_guester = "bad_bad"
   

    elif len(right_up_fingers)==2 and right_up_fingers[0]==8 and right_up_fingers[1]==12 and len(left_up_fingers) == 0:
        str_guester = "2"
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 8 and right_up_fingers[1] == 12  and left_up_fingers[0] == 8 and left_up_fingers[1] ==12:
        str_guester = "2_2"
        

    
        
    elif len(right_up_fingers)==2 and right_up_fingers[0]==4 and right_up_fingers[1]==8 and len(left_up_fingers) == 0:
        str_guester = "8"
    
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 4 and right_up_fingers[1] == 8  and left_up_fingers[0] == 8 and left_up_fingers[1] ==4:
        str_guester = "8_8"
    
    elif len(right_up_fingers) == 2 and right_up_fingers[0] == 4 and right_up_fingers[1] == 12 and len(left_up_fingers) == 0:
        str_guester = "bb"
    
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 4 and right_up_fingers[1] == 12 and left_up_fingers[0] == 4 and left_up_fingers[1] == 12:
        str_guester = "bb_bb"

    elif len(right_up_fingers) == 2 and right_up_fingers[0] == 12 and right_up_fingers[1] == 20 and len(left_up_fingers) == 0: 
        str_guester = "cc"
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 12 and right_up_fingers[1] == 20 and left_up_fingers[0] == 12 and left_up_fingers[1] == 20:
        str_guester = "cc_cc"
    
    elif len(right_up_fingers) == 2 and right_up_fingers[0] == 8 and right_up_fingers[1] == 20 and len(left_up_fingers) == 0:
        str_guester = "dd"
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 8 and right_up_fingers[1] == 20 and left_up_fingers[0] == 8 and left_up_fingers[1] == 20:
        str_guester = "dd_dd"

    elif len(right_up_fingers) == 2 and right_up_fingers[0] == 16 and right_up_fingers[1] == 20 and len(left_up_fingers) == 0:
        str_guester = "ee"
    elif len(right_up_fingers) == 2 and len(left_up_fingers) == 2 and right_up_fingers[0] == 16 and right_up_fingers[1] == 20 and left_up_fingers[0] == 16 and left_up_fingers[1] == 20:
        str_guester = "ee_ee"
    
    elif len(right_up_fingers)==3 and right_up_fingers[0]==8 and right_up_fingers[1]==12 and right_up_fingers[2]==16:
        str_guester = "3"
    
    elif len(right_up_fingers)==3 and len(left_up_fingers) == 3 and right_up_fingers[0]==8 and right_up_fingers[1]==12 and right_up_fingers[2]==16 and left_up_fingers[0]==8 and left_up_fingers[1]==12 and left_up_fingers[2]==16:
        str_guester = "3_3"
    
    elif len(right_up_fingers)==3 and right_up_fingers[0]==4 and right_up_fingers[1]==8 and right_up_fingers[2]==12:
  
        dis_8_12 = list_lms[8,:] - list_lms[12,:]
        dis_8_12 = np.sqrt(np.dot(dis_8_12,dis_8_12))
        
        dis_4_12 = list_lms[4,:] - list_lms[12,:]
        dis_4_12 = np.sqrt(np.dot(dis_4_12,dis_4_12))
        
        if dis_4_12/(dis_8_12+1) <3:
            str_guester = "7"
        
        elif dis_4_12/(dis_8_12+1) >5:
            str_guester = "Gun"
        else:
            str_guester = "7"
    elif len(right_up_fingers) == 3 and len(left_up_fingers) == 3 and right_up_fingers[0] == 4 and right_up_fingers[1] == 8  and right_up_fingers[2] == 12  and left_up_fingers[0] == 8 and left_up_fingers[1] ==4 and left_up_fingers[2] ==12:
        str_guester = "gun_gun"    
    # 29  


    elif len(right_up_fingers) == 3 and right_up_fingers[0] == 8 and right_up_fingers[1] == 16 and right_up_fingers[2] == 20 :
        str_guester = "ff"
    elif len(right_up_fingers) == 3 and len(left_up_fingers) == 3 and right_up_fingers[0] == 8 and right_up_fingers[1] == 16 and right_up_fingers[2] == 20 and left_up_fingers[0] == 8 and left_up_fingers[1] == 16 and left_up_fingers[2] == 20:
        str_guester = "ff_ff"
    
    elif len(right_up_fingers) == 3 and right_up_fingers[0] == 12 and right_up_fingers[1] == 16 and right_up_fingers[2] == 20:
        str_guester = "gg"
    elif len(right_up_fingers) == 3 and len(left_up_fingers) == 3 and right_up_fingers[0] == 12 and right_up_fingers[1] == 16 and right_up_fingers[2] == 20 and left_up_fingers[0] == 12 and left_up_fingers[1] == 16 and left_up_fingers[2] == 20:
        str_guester = "gg_gg"
    
    
    elif len(right_up_fingers) == 3 and right_up_fingers[0] == 4 and right_up_fingers[1] == 8 and right_up_fingers[2] == 20:
        str_guester = "hh"
    elif len(right_up_fingers) == 3 and len(left_up_fingers) == 3 and right_up_fingers[0] == 4 and right_up_fingers[1] == 8 and right_up_fingers[2] == 20 and left_up_fingers[0] == 4 and left_up_fingers[1] == 8 and left_up_fingers[2] == 20:
        str_guester = "hh_hh"
    
    
    elif len(right_up_fingers) == 3 and right_up_fingers[0] == 8 and right_up_fingers[1] == 12 and right_up_fingers[2] == 20:
        str_guester = "ii"
    elif len(right_up_fingers) == 3 and len(left_up_fingers) == 3 and left_up_fingers[0] == 8 and left_up_fingers[1] == 12 and left_up_fingers[2] == 20:
        str_guester = "ii_li"
    
    
    elif len(right_up_fingers)==4 and right_up_fingers[0]==8 and right_up_fingers[1]==12 and right_up_fingers[2]==16 and right_up_fingers[3]==20:
        str_guester = "4"
    elif len(right_up_fingers)==4 and right_up_fingers[0]==20 and right_up_fingers[1]==16 and right_up_fingers[2]==12 and right_up_fingers[3]==8:
        str_guester = "4_4"
 
    
    elif len(right_up_fingers)==5 and len(left_up_fingers) == 0:
        str_guester = "5"
        
    elif len(right_up_fingers)==2 and right_up_fingers[0]==4 and right_up_fingers[1]==20 and len(left_up_fingers) == 0:
        str_guester = "6"
    elif len(right_up_fingers)==2 and len(left_up_fingers) == 2 and right_up_fingers[0]==4 and right_up_fingers[1]==20 and left_up_fingers[0]==4 and left_up_fingers[1]==20:
        str_guester = "6_6"
        
    elif len(right_up_fingers)==0 and len(left_up_fingers) == 0:
        str_guester = "10"
    
    
    else:
        str_guester = " "
        
    return str_guester


def run(bytes_data):
    # frame_count = 0
    # max_gesture_count = 0
    # max_gesture = None
    # gesture_count = {}

    # 将视频二进制数据保存为临时文件
    # with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
    #     temp_file.write(bytes_data)
    #     print(temp_file.name)
    #     temp_file_path = temp_file.name

    # 使用临时文件路径作为视频源进行处理
    # time.sleep(3)
    # video_stream = cv2.VideoCapture(temp_file_path)
    
    video_stream = cv2.VideoCapture(0)
    # video_stream.open(''.join(map(chr, bytes_data)))
    
    # os.remove(temp_file_path)
    # cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    
    while True:
        
        # 读取一帧图像
        success, img = video_stream.read()
        if not success:
            print(2)
            break
        image_height, image_width, _ = np.shape(img)
        print(1)
        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 得到检测结果
        left_up_fingers = []
        results = hands.process(imgRGB)
        print(3)

        right_up_fingers = []

        if results.multi_hand_landmarks:
            # for hand in results.multi_hand_landmarks:
            for idx, hand in enumerate(results.multi_hand_landmarks): 
                hand_type = results.multi_handedness[idx].classification[0].label 
                # print(hand_type) # 打印手的类型（左手或右手）
                
                # 双手检测
                # hand = results.multi_hand_landmarks[0]
                # hand_type = results.multi_handedness[hand].classification[0].label
                print()

                if hand_type == 'Left':
                    

                    mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x*image_width
                        pos_y = hand.landmark[i].y*image_height
                        list_lms.append([int(pos_x),int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms,dtype=np.int32)
                    hull_index = [0,1,2,3,6,10,14,19,18,17,10]
                    hull = cv2.convexHull(list_lms[hull_index,:])
                    # 绘制凸包
                    cv2.polylines(img,[hull], True, (0, 255, 0), 2)

                    # 查找外部的点数-指尖的点
                    n_fig = -1
                    ll = [4,8,12,16,20]
                    # print("new"+"2")
                    


                    for i in ll:
                        pt = (int(list_lms[i][0]),int(list_lms[i][1]))
                        # 检测点是否在凸包的外面
                        dist= cv2.pointPolygonTest(hull,pt,True)
                        if dist <0:
                            left_up_fingers.append(i)

        

                    for i in ll:
                        pos_x = hand.landmark[i].x*image_width
                        pos_y = hand.landmark[i].y*image_height
                        # 画点
                        cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)
                    # print("new"+"5")

                    
                elif hand_type == 'Right':
                    
                    
                    mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x*image_width
                        pos_y = hand.landmark[i].y*image_height
                        list_lms.append([int(pos_x),int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms,dtype=np.int32)
                    hull_index = [0,1,2,3,6,10,14,19,18,17,10]
                    hull = cv2.convexHull(list_lms[hull_index,:])
                    # 绘制凸包
                    cv2.polylines(img,[hull], True, (0, 255, 0), 2)

                    # 查找外部的点数-指尖的点
                    n_fig = -1
                    ll = [4,8,12,16,20]

                    for i in ll:
                        pt = (int(list_lms[i][0]),int(list_lms[i][1]))
                        # 检测点是否在凸包的外面
                        dist= cv2.pointPolygonTest(hull,pt,True)
                        if dist <0:
                            right_up_fingers.append(i)

                    # print(up_fingers)
                    # print(list_lms)
                    # print(np.shape(list_lms))
                    # 凸包外点的list

    

   

            str_guester = get_str_guester(left_up_fingers,right_up_fingers,list_lms)
            # str_guester = 0

            return str_guester

    return '1111'
