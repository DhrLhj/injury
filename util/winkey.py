#
# _*_ coding:UTF-8 _*_
 
import win32api
import win32con
import win32gui
from ctypes import *
import time
import numpy as np
 
VK_CODE = {
    'backspace':0x08,
    'tab':0x09,
    'clear':0x0C,
    'enter':0x0D,
    'shift':0x10,
    'ctrl':0x11,
    'alt':0x12,
    'pause':0x13,
    'caps_lock':0x14,
    'esc':0x1B,
    'spacebar':0x20,
    'page_up':0x21,
    'page_down':0x22,
    'end':0x23,
    'home':0x24,
    'left_arrow':0x25,
    'up_arrow':0x26,
    'right_arrow':0x27,
    'down_arrow':0x28,
    'select':0x29,
    'print':0x2A,
    'execute':0x2B,
    'print_screen':0x2C,
    'ins':0x2D,
    'del':0x2E,
    'help':0x2F,
    '0':0x30,
    '1':0x31,
    '2':0x32,
    '3':0x33,
    '4':0x34,
    '5':0x35,
    '6':0x36,
    '7':0x37,
    '8':0x38,
    '9':0x39,
    'a':0x41,
    'b':0x42,
    'c':0x43,
    'd':0x44,
    'e':0x45,
    'f':0x46,
    'g':0x47,
    'h':0x48,
    'i':0x49,
    'j':0x4A,
    'k':0x4B,
    'l':0x4C,
    'm':0x4D,
    'n':0x4E,
    'o':0x4F,
    'p':0x50,
    'q':0x51,
    'r':0x52,
    's':0x53,
    't':0x54,
    'u':0x55,
    'v':0x56,
    'w':0x57,
    'x':0x58,
    'y':0x59,
    'z':0x5A,
    'numpad_0':0x60,
    'numpad_1':0x61,
    'numpad_2':0x62,
    'numpad_3':0x63,
    'numpad_4':0x64,
    'numpad_5':0x65,
    'numpad_6':0x66,
    'numpad_7':0x67,
    'numpad_8':0x68,
    'numpad_9':0x69,
    'multiply_key':0x6A,
    'add_key':0x6B,
    'separator_key':0x6C,
    'subtract_key':0x6D,
    'decimal_key':0x6E,
    'divide_key':0x6F,
    'F1':0x70,
    'F2':0x71,
    'F3':0x72,
    'F4':0x73,
    'F5':0x74,
    'F6':0x75,
    'F7':0x76,
    'F8':0x77,
    'F9':0x78,
    'F10':0x79,
    'F11':0x7A,
    'F12':0x7B,
    'F13':0x7C,
    'F14':0x7D,
    'F15':0x7E,
    'F16':0x7F,
    'F17':0x80,
    'F18':0x81,
    'F19':0x82,
    'F20':0x83,
    'F21':0x84,
    'F22':0x85,
    'F23':0x86,
    'F24':0x87,
    'num_lock':0x90,
    'scroll_lock':0x91,
    'left_shift':0xA0,
    'right_shift ':0xA1,
    'left_control':0xA2,
    'right_control':0xA3,
    'left_menu':0xA4,
    'right_menu':0xA5,
    'browser_back':0xA6,
    'browser_forward':0xA7,
    'browser_refresh':0xA8,
    'browser_stop':0xA9,
    'browser_search':0xAA,
    'browser_favorites':0xAB,
    'browser_start_and_home':0xAC,
    'volume_mute':0xAD,
    'volume_Down':0xAE,
    'volume_up':0xAF,
    'next_track':0xB0,
    'previous_track':0xB1,
    'stop_media':0xB2,
    'play/pause_media':0xB3,
    'start_mail':0xB4,
    'select_media':0xB5,
    'start_application_1':0xB6,
    'start_application_2':0xB7,
    'attn_key':0xF6,
    'crsel_key':0xF7,
    'exsel_key':0xF8,
    'play_key':0xFA,
    'zoom_key':0xFB,
    'clear_key':0xFE,
    '+':0xBB,
    ',':0xBC,
    '-':0xBD,
    '.':0xBE,
    '/':0xBF,
    '`':0xC0,
    ';':0xBA,
    '[':0xDB,
    '\\':0xDC,
    ']':0xDD,
    "'":0xDE,
    '`':0xC0}
 
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
    print(get_screen_xy())
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