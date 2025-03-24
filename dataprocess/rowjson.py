import json

# 输入 JSON 文件路径
input_path = "/root/course/llava/data/caption_key3_sim_bey25.json"

# 读取 JSON 文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 读取并解析 JSON 文件

# 获取 JSON 对象的数量
num_objects = len(data)

# 打印数量
print(f"The number of JSON objects: {num_objects}")
