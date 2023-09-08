import re

def find_and_convert(s):
    # 使用正则表达式查找'00'后的所有数字字符
    match = re.search('00(\d+)', s)
    
    # 如果找到匹配项，返回转换后的数字
    if match:
        return int(match.group(1))
    # 如果没有匹配项，返回None
    return None

s = "abc0010xyz"
number = find_and_convert(s)

if number is not None:
    print(number)
else:
    print("No number found after '00'.")
