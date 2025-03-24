from keybert import KeyBERT
import csv

# 初始化 KeyBERT 模型
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# 输入和输出文件路径
input_path = "/root/course/llava/data/cap_1000_new.txt"
output_path = "/root/course/llava/data/caption_key3.txt"

# 打开输入文件和输出文件
with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", newline="", encoding="utf-8") as f_out:

    # 创建CSV写入器
    writer = csv.writer(f_out)
    writer.writerow(["image_name", "comment", "keywords"])  # 写入表头

    reader = csv.reader(f_in)
    next(reader, None)  # 跳过表头
    print("start")
    # 逐行处理
    for row in reader:
        image_name = row[0].strip()
        comment = row[1].strip()

        # 提取关键词
        keywords = kw_model.extract_keywords(
            comment,
            keyphrase_ngram_range=(1, 2),  # n-gram范围设置为1和2
            stop_words="english",  # 去掉英文停用词
            use_maxsum=True,  # 使用最大边际相关性来选择关键词
            top_n=3  # 提取前3个关键词
        )

        # 提取关键词并格式化为逗号分隔的字符串
        keyword_list = [keyword for keyword, _ in keywords]

        # 写入文件
        writer.writerow([image_name, comment, ",".join(keyword_list)])

print("关键词提取完成，已保存至 caption_key3.txt")
