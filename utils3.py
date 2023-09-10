import pandas as pd

class GestureLookup:
    def __init__(self, xlsx_path):
        # 使用read_excel代替read_csv
        self.df = pd.read_excel(xlsx_path, engine='openpyxl',dtype=str)
        self.id_to_gesture = dict(zip(self.df.iloc[:, 0], self.df.iloc[:, 1]))

    def get_gesture_by_id(self, gesture_id):
        print( self.id_to_gesture)
        return self.id_to_gesture.get(gesture_id, ' ')

# 使用示例
lookup = GestureLookup(r'C:\Users\25352\Desktop\1\map.xlsx')
gesture_id = '1126'
gesture_string = lookup.get_gesture_by_id(gesture_id)
print(gesture_string)
