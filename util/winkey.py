#
# _*_ coding:UTF-8 _*_
 
import win32api
import win32con
import win32gui
from ctypes import *
import time
import numpy as np


def get_screen_dip():
    """
    获取屏幕的分辨率
    """
    return(win32api.GetSystemMetrics(win32con.SM_CXSCREEN),win32api.GetSystemMetrics(win32con.SM_CYSCREEN))


def getdim(arr_coords_saved):
    """
    鼠标移动转换
    """
    screen = get_screen_dip()
    xm = np.sqrt((arr_coords_saved[3,:]**2).sum())
    delt = (xm*2)/screen[1]
    ym = delt*(screen[0]/2)
    dx = screen[0]/2/xm
    dy = screen[1]/2/ym
    return xm,ym,dx,dy

# region 鼠标操作
class POINT(Structure):
    _fields_ = [("x", c_ulong),("y", c_ulong)]
def get_mouse_point():
    """
    获取鼠标位置,单位分辨率
    """
    po = POINT()
    windll.user32.GetCursorPos(byref(po))
    return int(po.x), int(po.y)

def mouse_move(x,y):
    """
    鼠标位置移动，单位分辨率
    """
    windll.user32.SetCursorPos(x, y)
    
def mouse_click(pos=None):
    """
    鼠标左单击；
    @param pos:给定需要单击的位置坐标，形如(x,y)，单位是分辨率。默认为None则不移动鼠标
    """
    if pos:
        mouse_move(*pos)
        time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
def mouse_rclick(pos=None):
    """
    鼠标右单击；
    @param pos:给定需要单击的位置坐标，形如(x,y)，单位是分辨率。默认为None则不移动鼠标
    """
    if pos:
        mouse_move(*pos)
        time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    
def mouse_dclick(pos=None):
    """
    鼠标左双击；
    @param pos:给定需要单击的位置坐标，形如(x,y)，单位是分辨率。默认为None则不移动鼠标
    """
    if pos:
        mouse_move(*pos)
        time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    
def mouse_move_coords(arr_coord,xm,ym,dx,dy):
    """
    根据坐标移动鼠标
    """
    x,y = arr_coord[3]
    if abs(x) > xm:
        x = x/abs(x+1e-6)*xm
    if abs(y) > ym:
        y = y/abs(y+1e-6)*ym
#     print(x,y)
    px = int((x+xm)*dx)
    py = int((ym-y)*dy)
#     print(px,py)
    mouse_move(px,py)
    time.sleep(0.001)
    
# endregion


def key_test(key_name="backspace",time_sleep=1):
    print(f"key name:{key_name},time sleep:{time_sleep}")


def key_input(key_name='esc',time_sleep=1):
    win32api.keybd_event(VK_CODE[key_name],0,0,0)
    win32api.keybd_event(VK_CODE[key_name],0,win32con.KEYEVENTF_KEYUP,0)
    time.sleep(time_sleep)
    
def key_input_ctrl(key_name="backspace",time_sleep=1):
    win32api.keybd_event(0x11, 0, 0, 0)
    win32api.keybd_event(VK_CODE[key_name], 0, 0, 0)
    win32api.keybd_event(VK_CODE[key_name], 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)


def test():
    print(get_screen_dip())
    time.sleep(5)
    list_keys = ["left_arrow","a","up_arrow",'b',"right_arrow",'c',"down_arrow",'d',"home",'1',"end",'9']
    for key in list_keys:
        key_input(key)

    key_input_ctrl("s")


if __name__ == "__main__":
    # t4()
    # t3()
    #t2()
    t4()
    # t0()
