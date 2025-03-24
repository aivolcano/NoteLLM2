import csv

# 输入和输出文件路径
input_path = "/root/course/llava/data/caption_key3_sim.txt"
output_path = "/root/course/llava/data/caption_key3_sim_bey25.txt"

# 读取并过滤数据
filtered_results = []

with open(input_path, "r", encoding="utf-8") as f_in:
    reader = csv.reader(f_in)
    header = next(reader)  # 读取表头

    for row in reader:
        if len(row) < 5:
            continue

        # 提取数据
        image_name = row[0].strip()
        comment = row[1].strip()
        keywords = row[2].strip()
        sim_picture = row[3].strip()
        sim_score = float(row[4].strip())  # 转换为浮动类型

        # 如果相似度小于 0.25，则跳过
        if sim_score >= 0.25:
            filtered_results.append(row)

# 将结果保存到新的文件
with open(output_path, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header)  # 写入表头
    for row in filtered_results:
        writer.writerow(row)

print(f"Done! Filtered results have been saved to: {output_path}")
