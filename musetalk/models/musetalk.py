import json

import cv2
import sys
import torch
import numpy as np
from torch import nn
from diffusers import UNet2DConditionModel, AutoencoderKL

sys.path.append('.')
from common.setting import UNET_CONFIG_PATH, VAE_PATH


class MuseTalkNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        with open(UNET_CONFIG_PATH, "r") as f:
            unet_config = json.load(f)
        self.unet = UNet2DConditionModel(**unet_config)

    def forward(self, images, audios):
        return self.unet(images, 0, encoder_hidden_states=audios).sample


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
    vae.requires_grad_(False)
    mt = MuseTalkNetwork().to(device)
    # 准备数据，形状为batch_size， 8, 32, 32
    # latent_model_input = torch.cat([masked_latents, related_latents], dim=1)
    image = cv2.imread("./datasets/images/tjl/0.png")
    image = cv2.resize(image, (256, 256))
    image = torch.FloatTensor(np.transpose(image / 255., (2, 0, 1)))

    image = image.to(device)

    audio = np.load('./datasets/audios/tjl/0.npy')

    mask = torch.zeros((image.shape[1], image.shape[2])).to(device)
    mask[:image.shape[1] // 2, :] = 1
    masked_image = image * mask

    latents = vae.encode(image[None]).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    masked_latents = vae.encode(masked_image[None]).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    latent_model_input = torch.cat([masked_latents, latents], dim=1)

    print(latent_model_input.shape)
    print(audio.shape)

    out_latents = mt(latent_model_input.to(device), torch.FloatTensor(audio[None]).to(device))
    outputs = vae.decode(latents).sample
    outputs = (outputs / 2 + 0.5).clamp(0, 1)
    print(outputs.shape, )
    outputs = outputs[0].detach().cpu().permute(1, 2, 0).float().numpy()
    outputs = (outputs * 255).round().astype("uint8")
    from PIL import Image
    #
    Image.fromarray(outputs).show()
