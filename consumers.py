import os

from channels.generic.websocket import WebsocketConsumer,AsyncWebsocketConsumer
from channels.exceptions import StopConsumer
from asgiref.sync import async_to_sync
from app01.newFigure import run,get_str_guester
from app01.sxzy import analyze_hand_gesture
from app01.detact_gestery import *

from threading import Thread 

class HandGestureDetector():
    def __init__(self):
        self.gesture_detector = GestureRecognition()
        self.cap = cv.VideoCapture(1)
        self.mode = 0
        self.number = -1
        self.gesture_id = -1
        
    def run(self):
        while True:
            # print('ready show...')
            ret, image = self.cap.read()
            image = cv.flip(image, 1)  # Mirror display
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            show_image, self.gesture_id = self.gesture_detector.recognize(image, self.number, self.mode)
            # if self.gesture_id!=-1:
            #     print(self.gesture_id)
            cv.imshow('Character writing', show_image)
            if cv.waitKey(1) == ord('q'):  # 点击视频，输入q退出
                break

class ChatConsumer(AsyncWebsocketConsumer):
    async def websocket_connect(self,meassge):
        print("有人来连接了.....")
        # 有客户端来向后端发送websocket连接的请求时，自动触发
        # 服务器端允许和客户端创建连接。
        # 接收客户端连接
        await self.accept()
        
        # init the hand gesture detector 
        self.handGestureDetector = HandGestureDetector()
        Thread(target = self.handGestureDetector.run).start()
        

        # 给客户端发送消息
        await self.send("hello,来了呀")

    async def receive(self, text_data=None, bytes_data=None):

        # print(run(bytes_data))
        # byte_array = bytearray(bytes_data)
        # file_path = os.path.join('./video', 'uploaded_video.mp4')
        # #
        # with open(file_path, 'wb+') as file:
        #     file.write(byte_array)

        # res = run(bytes_data=bytes_data)
        # result = analyze_hand_gesture(bytes_data=bytes_data)
        # if self.handGestureDetector.gesture_id!=-1:
        # await self.send("已经收到视频并解析")
        await self.send(str(self.handGestureDetector.gesture_id))
        self.handGestureDetector.gesture_id=-1
        # print(str(self.handGestureDetector.gesture_id))


    async def disconnect(self, close_code):
        #客户端与服务器断开连接时，自动触发。
        print("断开连接")
        # 关闭并保存文件
        raise StopConsumer()
