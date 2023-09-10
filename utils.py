from collections import deque, Counter
import copy
import itertools
import pandas as pd
class FixedSizeQueue:
    def __init__(self, size):
        self.size = size
        self.queue = deque(maxlen=size)

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        return self.queue.popleft() if len(self.queue) > 0 else None

    def get_all(self):
        return list(self.queue)

    def __len__(self):
        return len(self.get_all())

    def full(self):
        return len(self.queue) == self.size

    def most_common_in_last_n(self, n=30):
    # 获取最后n个元素
        last_n_items = list(self.queue)[-n:]
        last_n_items=[i.item() for i in last_n_items if i is not None]
        last_five_items=list(self.queue)[-5:]
        if None in last_five_items:
            return None,None
        # 使用Counter统计元素出现的次数
        count = Counter(last_n_items)
        # 获取出现次数最多的元素
        most_common = count.most_common(1)
        print("times",most_common)
        if most_common and most_common[0][1] > 12:
            return most_common[0]  # Returns (value, count) tuple
        return None, None
    
       # 清空队列的方法
    def clear(self):
        self.queue.clear()


import pyautogui
import time

class GestureLookup:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, dtype=str)  # 确保读取的列都是字符串格式
        self.id_to_gesture = dict(zip(self.df.iloc[:, 0], self.df.iloc[:, 1]))

    def get_gesture_by_id(self, gesture_id):
        return self.id_to_gesture.get(str(gesture_id), ' ')  # 在查询之前将gesture_id转换为字符串

def press_keys_from_string(s: str):
    """
    Simulate pressing keys from a given string at the same time.

    Args:
    - s (str): The string containing keys to be pressed simultaneously.

    Note:
    This function will press and hold down the keys in the order they appear in the string, 
    and then release them in the same order.
    """
    s=[tem for tem in s if tem !=' ']
    if s!=[]
        pyautogui.keyDown('ctrl')
        for char in s:
            pyautogui.keyDown(char)
        for char in s:
            pyautogui.keyUp(char)
        pyautogui.keyUp('ctrl')
        print(s)




def calc_landmark_list( image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

def pre_process_landmark( landmark_list):
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



import torch

def average_movement_vector(coord_t1: torch.Tensor, coord_t2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the average movement vector between two sets of coordinates.

    Parameters:
    - coord_t1 : torch.Tensor : The coordinates at time t1, shape [21, 2]
    - coord_t2 : torch.Tensor : The coordinates at time t2, shape [21, 2]

    Returns:
    - torch.Tensor : The average movement vector, shape [1, 2]
    """
    if coord_t1.shape != (21, 2) or coord_t2.shape != (21, 2):
        raise ValueError("The shape of both input tensors must be [21, 2]")

    # Calculate the movement vectors for each node
    movement_vectors = coord_t2 - coord_t1

    # Compute the average movement vector
    avg_vector = torch.mean(movement_vectors, axis=0, keepdim=True).reshape(2)

    return avg_vector




# 使用示例
if __name__=='__main__':
    queue = FixedSizeQueue(60)

    # 添加元素到队列
    for i in range(50):
        queue.push(i)

    for i in range(10):
        queue.push(100)  # 添加10次数值为100的元素

    # 获取最近30个元素中出现次数最多的值
    result = queue.most_common_in_last_n(30)
    print(result)  # 输出：(100, 10)，因为100出现了10次

    # 示例:
    coord_t1 = torch.rand(21, 2)
    coord_t2 = torch.rand(21, 2)
    print(average_movement_vector(coord_t1, coord_t2))
