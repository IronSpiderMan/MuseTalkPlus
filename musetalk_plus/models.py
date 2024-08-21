import json
from pathlib import Path

import torch
from torch import nn
from diffusers import UNet2DConditionModel


class MuseTalkModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        config_path = Path(model_path) / 'config.json'
        ckpt_path = Path(model_path) / 'diffusion_pytorch_model.bin'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.unet = UNet2DConditionModel(**config)
        weights = torch.load(ckpt_path)
        self.unet.load_state_dict(weights)

        # self.unet = UNet2DConditionModel.from_pretrained(
        #     model_path, use_safetensors=False,
        #     low_cpu_mem_usage=False,  # 不使用低内存模式
        #     device_map=None  # 不使用设备映射
        # )

    def forward(self, inputs):
        image_feature, audio_feature = inputs
        return self.unet(image_feature, 0, encoder_hidden_states=audio_feature).sample
