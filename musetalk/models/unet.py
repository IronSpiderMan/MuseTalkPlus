import sys
import math
import json

import cv2
import torch
import numpy as np
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

sys.path.append('.')
from musetalk.models.vae import VAE
from common.setting import UNET_MODEL_PATH, UNET_CONFIG_PATH


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


class UNet:
    def __init__(
            self,
            unet_config,
            model_path,
            use_float16=False,
    ):
        with open(unet_config, 'r') as f:
            unet_config = json.load(f)
        self.model = UNet2DConditionModel(**unet_config)
        self.pe = PositionalEncoding(d_model=384)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(model_path).endswith('.safetensors'):
            weights = {}
            with safe_open(model_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,
                                                                                          map_location=self.device)
        self.model.load_state_dict(weights)
        if use_float16:
            self.model = self.model.half()
        self.model.to(self.device)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pe = PositionalEncoding().to(device)

    unet = UNet(UNET_CONFIG_PATH, UNET_MODEL_PATH)
    unet.model = unet.model.to(device)
    vae = VAE()
    vae.vae = vae.vae.to(device)
    vae.vae.requires_grad_(False)

    fidx = 300

    # 准备数据，形状为batch_size， 8, 32, 32
    # image = cv2.imread(f"./datasets/images/tjl1/{fidx}.png")
    image = cv2.imread(f"./300.png")
    image = cv2.resize(image, (256, 256))
    audio = np.load(f'./datasets/audios/tjl/{fidx}.npy')
    audio = pe(torch.FloatTensor(audio[None]).to(device))
    latents = vae.get_latents_for_unet(image)
    out_latents = unet.model(latents, 0, encoder_hidden_states=audio).sample
    outputs = vae.decode_latents(out_latents)[0][:, :, ::-1]

    from PIL import Image

    Image.fromarray(outputs).save('1.jpg')
