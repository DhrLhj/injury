import pandas as pd

class GestureLookup:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.id_to_gesture = dict(zip(self.df.iloc[:, 0], self.df.iloc[:, 1]))

    def get_gesture_by_id(self, gesture_id):
        return self.id_to_gesture.get(gesture_id, None)

# 使用示例
lookup = GestureLookup(r'C:\Users\25352\Desktop\map.csv')
gesture_id = '11'  # 例如，假设你要查找的手势id为1
gesture_string = lookup.get_gesture_by_id(gesture_id)
if gesture_string:
    print(f"The gesture string for ID {gesture_id} is: {gesture_string}")
else:
    print(f"No gesture string found for ID {gesture_id}")
