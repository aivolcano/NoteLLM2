from transformers import LlavaForConditionalGeneration
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
# from transformers.modeling_outputs import CausalLMOutputWithPast

class CustomLlavaModel(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以添加自定义的初始化代码

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     pixel_values: torch.FloatTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     vision_feature_layer: Optional[int] = None,
    #     vision_feature_select_strategy: Optional[str] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     num_logits_to_keep: int = 0,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     r"""
    #     Args:
    #         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    #         num_logits_to_keep (`int`, *optional*):
    #             Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
    #             `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
    #             token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


    #     Returns:

    #     Example:

    #     ```python
    #     >>> from PIL import Image
    #     >>> import requests
    #     >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    #     >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    #     >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    #     >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    #     >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    #     >>> image = Image.open(requests.get(url, stream=True).raw)

    #     >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    #     >>> # Generate
    #     >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    #     >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    #     ```"""

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     vision_feature_layer = (
    #         vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    #     )
    #     vision_feature_select_strategy = (
    #         vision_feature_select_strategy
    #         if vision_feature_select_strategy is not None
    #         else self.config.vision_feature_select_strategy
    #     )

    #     if (input_ids is None) ^ (inputs_embeds is not None):
    #         raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    #     if pixel_values is not None and inputs_embeds is not None:
    #         raise ValueError(
    #             "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
    #         )

    #     if inputs_embeds is None:
    #         inputs_embeds = self.get_input_embeddings()(input_ids)

    #     if pixel_values is not None:
    #         image_features = self.get_image_features(
    #             pixel_values=pixel_values,
    #             vision_feature_layer=vision_feature_layer,
    #             vision_feature_select_strategy=vision_feature_select_strategy,
    #         )

    #         n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
    #         n_image_features = image_features.shape[0] * image_features.shape[1]
    #         if n_image_tokens != n_image_features:
    #             raise ValueError(
    #                 f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    #             )
    #         special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
    #         special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    #         image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    #         inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    #     return super().forward(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         vision_feature_layer=vision_feature_layer,
    #         vision_feature_select_strategy=vision_feature_select_strategy,
    #         labels=labels,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #         num_logits_to_keep=num_logits_to_keep,
    #     )

    def extract_embedding(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取文本和图像的embedding

        Args:
            input_ids: 输入的token ids
            pixel_values: 输入的图像
            attention_mask: 注意力掩码
            position_ids: 位置编码

        Returns:
            tuple: (text_embedding, image_embedding)
                - text_embedding: 文本的最后一个token的embedding
                - image_embedding: 图像的embedding
        """
        # 设置默认参数
        output_hidden_states = True  # 需要hidden states来获取embedding
        return_dict = True

        # 获取输入的embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 处理图像特征
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )

            # 找到图像token的位置
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # 记录最后一个图像token的位置
            image_positions = (input_ids == self.config.image_token_index).nonzero()
            if len(image_positions) > 0:
                last_image_position = image_positions[-1][1].item()
            else:
                last_image_position = None

        # 获取模型输出
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取文本的最后一个token的embedding
        hidden_states = outputs.hidden_states[-1]  # 使用最后一层的hidden states
        text_embedding = hidden_states[:, -1, :]  # 获取最后一个位置的embedding

        # 获取图像的embedding
        if pixel_values is not None and last_image_position is not None:
            image_embedding = hidden_states[:, last_image_position, :]
        else:
            image_embedding = None

        return text_embedding, image_embedding

    def micl_loss(self, q_i, q_t, t_i, t_t, temperature=0.07, alpha=0.1):
        """
        计算MICL (Multimodal In-Context Learning)损失,使用late fusion机制
        """
        # 确保输入张量都有梯度
        q_i = q_i.detach().clone().requires_grad_(True)
        q_t = q_t.detach().clone().requires_grad_(True)
        t_i = t_i.detach().clone().requires_grad_(True)
        t_t = t_t.detach().clone().requires_grad_(True)

        # 对embeddings进行L2归一化
        q_i = nn.functional.normalize(q_i, dim=-1)
        q_t = nn.functional.normalize(q_t, dim=-1)
        t_i = nn.functional.normalize(t_i, dim=-1)
        t_t = nn.functional.normalize(t_t, dim=-1)

        batch_size = q_i.shape[0]
        hidden_dim = q_i.shape[1]

        # 检查数值
        if torch.isnan(q_i).any() or torch.isnan(q_t).any() or torch.isnan(t_i).any() or torch.isnan(t_t).any():
            return torch.tensor(0.0, device=q_i.device, dtype=q_i.dtype, requires_grad=True)

        # Late Fusion参数 (如果还没有,需要创建)
        if not hasattr(self, 'fusion_layer'):
            dtype = q_i.dtype
            device = q_i.device
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype).to(device)
            self.fusion_bias = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=dtype, device=device), requires_grad=True)

        # 添加数值稳定性
        eps = 1e-6

        # Query的视觉-文本融合
        fusion_input_q = torch.cat([q_i, q_t], dim=-1)
        fusion_output = self.fusion_layer(fusion_input_q)
        z_q = torch.sigmoid(fusion_output + self.fusion_bias)
        q_fused = z_q * q_i + (1 - z_q) * q_t

        # Target的视觉-文本融合
        fusion_input_t = torch.cat([t_i, t_t], dim=-1)
        fusion_output_t = self.fusion_layer(fusion_input_t)
        z_t = torch.sigmoid(fusion_output_t + self.fusion_bias)
        t_fused = z_t * t_i + (1 - z_t) * t_t

        # 计算视觉对比损失 Lv
        sim_i2i = torch.matmul(q_i, t_i.T) / (temperature + eps)
        sim_i2i = torch.clamp(sim_i2i, min=-1e4, max=1e4)  # 防止数值溢出
        labels = torch.arange(batch_size, device=q_i.device)

        # 如果batch_size=1,添加一个虚拟的负样本
        if batch_size == 1:
            neg_sample = torch.tensor([[-10.0]], device=q_i.device, dtype=q_i.dtype, requires_grad=True)
            sim_i2i = torch.cat([sim_i2i, neg_sample], dim=1)
            labels = torch.zeros(1, device=q_i.device, dtype=torch.long)

        visual_loss = nn.functional.cross_entropy(sim_i2i, labels)

        # 计算多模态对比损失 Lm
        sim_fusion = torch.matmul(q_fused, t_fused.T) / (temperature + eps)
        sim_fusion = torch.clamp(sim_fusion, min=-1e4, max=1e4)

        # 同样处理batch_size=1的情况
        if batch_size == 1:
            neg_sample = torch.tensor([[-10.0]], device=q_i.device, dtype=q_i.dtype, requires_grad=True)
            sim_fusion = torch.cat([sim_fusion, neg_sample], dim=1)

        multimodal_loss = nn.functional.cross_entropy(sim_fusion, labels)

        # 计算总损失并添加梯度裁剪
        total_loss = (visual_loss + alpha * multimodal_loss) / (1 + alpha)

        return total_loss

