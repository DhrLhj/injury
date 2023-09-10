import pyautogui

pyautogui.FAILSAFE = False
# =================
# pyautogui xOy:
# O->>>>>>>>>>>>>>>X
# v
# v
# Y
# just like media-pipe

SIZE_SCREEN_WIDTH, SIZE_SCREEN_HEIGHT = pyautogui.size()

def mouse_move(x_delta,y_delta):
    # first move to the center of the screen
    # pyautogui.moveTo(SIZE_SCREEN_WIDTH//2, SIZE_SCREEN_HEIGHT//2, duration=0.1)
    # move according the x,y
    X_DELTA = int(x_delta/10.0*SIZE_SCREEN_WIDTH)
    Y_DELTA = int(y_delta/10.0*SIZE_SCREEN_HEIGHT)
    pyautogui.moveRel(X_DELTA, Y_DELTA, duration=0)


if __name__ == "__main__":
    mouse_move(0.2,0.1)
