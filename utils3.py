import pandas as pd

class GestureLookup:
    def __init__(self, xlsx_path):
        # 使用read_excel代替read_csv
        self.df = pd.read_excel(xlsx_path, engine='openpyxl',dtype=str)
        self.id_to_gesture = dict(zip(self.df.iloc[:, 0], self.df.iloc[:, 1]))

    def get_gesture_by_id(self, gesture_id):
        # print( self.id_to_gesture)
        return self.id_to_gesture.get(gesture_id, ' ')

# 使用示例
lookup = GestureLookup(r'C:\Users\25352\Desktop\工作簿1.xlsx')
gesture_id = '0620'
gesture_string = lookup.get_gesture_by_id(gesture_id)
if gesture_string.strip():
    print(f"The gesture string for ID {gesture_id} is: {gesture_string}")
else:
    print(f"No gesture string found for ID {gesture_id}")
