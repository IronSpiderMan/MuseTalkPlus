import torch
from torch import nn
import numpy as np
from diffusers import UNet2DConditionModel


class MuseTalkModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # self.ape = nn.Embedding(50, 384)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, use_safetensors=False,
        )

    def forward(self, inputs):
        image_feature, audio_feature = inputs
        # b, seq_len, dim = audio_feature.shape
        # indexes = torch.LongTensor(np.arange(seq_len))
        # indexes = indexes.repeat(b, 1).to(audio_feature.device)
        # audio_feature = self.ape(indexes) + audio_feature
        return self.unet(image_feature, 0, encoder_hidden_states=audio_feature).sample
