# NoteLLM2

NoteLLM2是一个基于LLaVA架构的多模态大语言模型项目，专注于图像理解和文本生成任务。

## 项目结构
notellm2/
├── data/ # 数据目录
│ └── caption_key3_sim_bey25.json # 图像标注数据
├── dataprocess/ # 数据预处理
├── custom_llava.py # 模型架构定义
└── train_214.py # 训练脚本


## 注意事项
- 确保有足够的GPU内存进行训练 >24g
