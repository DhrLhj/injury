import cv2 as cv

from utils import CvFpsCalc
from gestures import  GestureRecognition




def main():
    gesture_detector = GestureRecognition()

    cap = cv.VideoCapture(0)
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1

    while True:
        fps = cv_fps_calc.get()
        ret, image = cap.read()

        debug_image,show_image, gesture_id = gesture_detector.recognize(image, number, mode)
        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)
        print(gesture_id)
        print(debug_image.shape)
        cv.imshow('Gesture Recognition', debug_image)
        cv.imshow('Character writing', show_image)
        if cv.waitKey(1) == ord('q'):  # 点击视频，输入q退出
            break

    cv.destroyAllWindows()





if __name__ == '__main__':
    main()