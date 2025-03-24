import csv
from sentence_transformers import SentenceTransformer, util

# 初始化 Sentence-BERT 模型
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 输入和输出文件路径
input_path = "/root/course/llava/data/caption_key3.txt"
output_path = "/root/course/llava/data/caption_key3_sim.txt"

# 存储图片关键词和嵌入向量
image2keys = {}
image2embed = {}

# 读取输入文件并处理数据
with open(input_path, "r", encoding="utf-8") as f_in:
    reader = csv.reader(f_in)
    next(reader, None)  # 跳过表头

    for row in reader:
        image_name = row[0].strip()
        comment = row[1].strip()
        keywords = row[2].strip()

        # 将图片的关键词和嵌入向量存储
        image2keys[image_name] = (comment, keywords.split(","))

        # 将关键词转换为向量
        keyword_embeddings = model.encode(image2keys[image_name][1])
        image2embed[image_name] = keyword_embeddings

# 计算两张图片关键词嵌入之间的余弦相似度
def calc_image_similarity(imgA, imgB):
    embedsA = image2embed[imgA]
    embedsB = image2embed[imgB]

    # 计算关键词嵌入的余弦相似度
    sim_score = util.cos_sim(embedsA, embedsB).mean().item()
    return sim_score

# 计算每张图片与其他图片之间的相似度，并找出最相似的图片
results = []
image_names = list(image2keys.keys())

for i in range(len(image_names)):
    imgA = image_names[i]
    best_match = None
    best_score = -1.0

    for j in range(len(image_names)):
        if i == j:
            continue
        imgB = image_names[j]
        score = calc_image_similarity(imgA, imgB)
        if score > best_score:
            best_score = score
            best_match = imgB

    # 获取当前图片的评论和关键词
    comment, keywords = image2keys[imgA]

    # 保存结果
    results.append([imgA, comment, ",".join(keywords), best_match, best_score])

# 将结果保存到输出文件
with open(output_path, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["image_name", "comment", "keywords", "sim_picture", "sim_score"])  # 写入表头
    for row in results:
        writer.writerow(row)

print(f"Done! Results have been saved to: {output_path}")
