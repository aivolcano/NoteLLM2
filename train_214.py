# 首先导入基础包
import os
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset


# 1. 导入所需的包
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from modelscope import snapshot_download
from custom_llava import CustomLlavaModel
from micl_model import MICLModel


class MICLDataset(Dataset):
    """MICL数据集类"""
    def __init__(self, json_path, image_dir, transform=None):
        """
        初始化数据集
        Args:
            json_path: JSON文件路径
            image_dir: 图片目录路径
            transform: 图像转换函数
        """
        self.data = self.load_data(json_path, image_dir)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def load_data(json_path, image_dir):
        """加载数据集"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = []
        for item in data:
            q=[]
            image_path = os.path.join(image_dir, item['query']['image_name'])
            if os.path.exists(image_path):
                q.append({
                    'image': image_path,
                    'topic': item['query']['keywords'],
                    'comment': item['query']['comment']
                })
            t=[]
            image_path = os.path.join(image_dir, item['target']['image_name'])
            if os.path.exists(image_path):
                t.append({
                    'image': image_path,
                    'topic': item['target']['keywords'],
                    'comment': item['target']['comment']
                })
            dataset.append({
                'query': q,
                'target': t
            })
        return dataset

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        item = self.data[idx]

        # 处理查询图像
        query_image = self.transform(item['query'][0]['image'])
        query_topic = item['query'][0]['topic']
        query_comment = item['query'][0]['comment']

        # 处理目标图像
        target_image = self.transform(item['target'][0]['image'])
        target_topic = item['target'][0]['topic']
        target_comment = item['target'][0]['comment']

        return {
            'query_image': query_image,
            'query_topic': query_topic,
            'query_comment': query_comment,
            'target_image': target_image,
            'target_topic': target_topic,
            'target_comment': target_comment
        }

from PIL import Image
class DataProcessor(nn.Module):
    """
    数据处理器类，用于处理MICL数据集的图像和文本
    结合了AutoProcessor的功能来处理多模态输入
    """
    def __init__(self, dataset, processor, device='cuda'):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.device = device  # 添加设备参数
                # 设置处理器的patch_size
        self.processor.patch_size = 196  # LLaVA默认使用14x14的patch size
        self.processor.num_additional_image_tokens = 1
        self.processor.vision_feature_select_strategy = "default"
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        item = self.dataset[idx]

        # 打开图像文件
        query_image = Image.open(item['query'][0]['image']).convert('RGB')
        target_image = Image.open(item['target'][0]['image']).convert('RGB')

        # 处理更长的prompt
        prompt_q = f"Note content: {{'topic': '{item['query'][0]['topic']}', 'content': '{item['query'][0]['comment']}', 'image': <image>}}, Compress this note into one word："

        # 使用更大的max_length处理输入
        input_q = self.processor(
            images=query_image,
            text=prompt_q,
            return_tensors="pt",
            padding='max_length',
            max_length=1024,  # 使用更大的max_length
            truncation=True
        )

        # 构建目标提示
        prompt_t = f"Note content: {{'topic': '{item['target'][0]['topic']}', 'content': '{item['target'][0]['comment']}', 'image': <image>}}, Compress this note into one word："

        # 处理目标输入
        input_t = self.processor(
            images=target_image,
            text=prompt_t,
            return_tensors="pt",
            padding=True
        )

        # 移除批次维度
        for k in input_q.keys():
            if torch.is_tensor(input_q[k]):
                input_q[k] = input_q[k].squeeze(0)
        for k in input_t.keys():
            if torch.is_tensor(input_t[k]):
                input_t[k] = input_t[k].squeeze(0)

        return {
            'query_inputs': input_q,
            'target_inputs': input_t
        }

    def collate_fn(self, batch):
        """
        将多个样本组合成一个批次
        Args:
            batch: 样本列表
        Returns:
            批处理后的数据
        """
        # 初始化批次数据结构
        batch_data = {
            'query_inputs': {
                'input_ids': [],
                'attention_mask': [],
                'pixel_values': []
            },
            'target_inputs': {
                'input_ids': [],
                'attention_mask': [],
                'pixel_values': []
            }
        }

        # 首先收集所有序列长度
        max_length = {
            'query_inputs': {'input_ids': 0, 'attention_mask': 0},
            'target_inputs': {'input_ids': 0, 'attention_mask': 0}
        }

        # 找出最大长度
        for sample in batch:
            for input_type in ['query_inputs', 'target_inputs']:
                max_length[input_type]['input_ids'] = max(
                    max_length[input_type]['input_ids'],
                    len(sample[input_type]['input_ids'])
                )
                max_length[input_type]['attention_mask'] = max(
                    max_length[input_type]['attention_mask'],
                    len(sample[input_type]['attention_mask'])
                )

        # 收集并padding数据
        for sample in batch:
            for input_type in ['query_inputs', 'target_inputs']:
                # 处理input_ids
                input_ids = sample[input_type]['input_ids']
                padding_length = max_length[input_type]['input_ids'] - len(input_ids)
                padded_input_ids = torch.cat([
                    input_ids,
                    torch.zeros(padding_length, dtype=input_ids.dtype)
                ])
                batch_data[input_type]['input_ids'].append(padded_input_ids)

                # 处理attention_mask
                attention_mask = sample[input_type]['attention_mask']
                padding_length = max_length[input_type]['attention_mask'] - len(attention_mask)
                padded_attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
                batch_data[input_type]['attention_mask'].append(padded_attention_mask)

                # 处理pixel_values (图像特征通常已经是固定大小)
                batch_data[input_type]['pixel_values'].append(sample[input_type]['pixel_values'])

        # 堆叠张量并移动到指定设备
        for input_type in ['query_inputs', 'target_inputs']:
            for key in batch_data[input_type].keys():
                batch_data[input_type][key] = torch.stack(batch_data[input_type][key]).to(self.device)

        return batch_data

def get_args():
    """配置训练参数"""
    args = {
        # 数据相关
        'json_path': 'data/caption_key3_sim_bey25.json',
        'image_dir': '/root/course/llava/data/img200/',
        'model_dir': 'swift/llava-1.5-7b-hf',

        # 训练相关
        'batch_size': 2,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'max_grad_norm': 1.0,
        'num_workers': 0,
        'sample_size': 15,  # 用于测试的样本数量

        # 模型配置
        'torch_dtype': torch.float16,
        'device_map': "auto",
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 添加实际设备参数
    }
    return args

def setup_model(args):
    """配置和初始化模型"""
    # 清空显存缓存
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

    # 量化配置
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 下载并加载模型
    model_dir = snapshot_download(args['model_dir'])
    model = CustomLlavaModel.from_pretrained(
        model_dir,
        quantization_config=nf4_config,
        device_map=args['device_map'],
        torch_dtype=args['torch_dtype']
    )

    # 设置模型训练模式
    model.train()

    # 配置参数梯度
    for name, param in model.named_parameters():
        param.requires_grad = param.dtype in [torch.float16, torch.float32, torch.float64]

    # 特别配置fusion_layer
    if hasattr(model, 'fusion_layer'):
        for param in model.fusion_layer.parameters():
            if param.dtype in [torch.float16, torch.float32, torch.float64]:
                param.requires_grad = True
        if hasattr(model, 'fusion_bias'):
            model.fusion_bias.requires_grad = True

    return model

def prepare_data(args, processor):
    """准备数据加载器"""
    # 加载数据集
    dataset = MICLDataset.load_data(args['json_path'], args['image_dir'])
    dataset = dataset[:args['sample_size']]  # 截取测试样本

    # 创建数据处理器，使用实际设备而不是device_map
    data_processor = DataProcessor(
        dataset=dataset,
        processor=processor,
        device=args['device']  # 使用新添加的device参数
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        data_processor,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=data_processor.collate_fn,
        num_workers=args['num_workers'],
        pin_memory=False
    )

    return train_loader

def train_epoch(model, train_loader, optimizer, args):
    """训练一个epoch"""
    total_loss = 0

    for batch in train_loader:
        # 获取输入
        query_inputs = batch['query_inputs']
        target_inputs = batch['target_inputs']

        # 提取嵌入
        q_i, q_t = model.extract_embedding(
            input_ids=query_inputs['input_ids'],
            pixel_values=query_inputs['pixel_values'],
            attention_mask=query_inputs['attention_mask']
        )

        t_i, t_t = model.extract_embedding(
            input_ids=target_inputs['input_ids'],
            pixel_values=target_inputs['pixel_values'],
            attention_mask=target_inputs['attention_mask']
        )

        # 计算MICL损失
        loss = model.micl_loss(q_i, q_t, t_i, t_t)

        # 检查梯度
        if not any(p.requires_grad for p in model.parameters()):
            print("警告: 没有参数需要梯度!")
            continue

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['max_grad_norm'])

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def train(args=None):
    """主训练函数"""
    if args is None:
        args = get_args()

    # 设置模型
    model = setup_model(args)
    model_dir = snapshot_download(args['model_dir'])
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_dir,
        use_fast=True,
    )

    # 准备数据
    train_loader = prepare_data(args, processor)

    # 配置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])

    # 训练循环
    for epoch in range(args['num_epochs']):
        avg_loss = train_epoch(model, train_loader, optimizer, args)
        print(f"Epoch {epoch+1}/{args['num_epochs']}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()
