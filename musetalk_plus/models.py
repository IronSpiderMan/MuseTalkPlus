import torch
from torch import nn
import numpy as np
from diffusers import UNet2DConditionModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x


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
