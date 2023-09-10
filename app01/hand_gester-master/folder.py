import os

def get_direct_paths(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)]

directory = r'C:\Users\25352\Desktop\hand-gester\handpose_x_gesture_v1'
direct_paths = get_direct_paths(directory)

for path in direct_paths:
    print(path)
