import os
import sys
import json

sys.path.append('..')

import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel

from train.musetalk.datasets import MuseTalkDataset
from common.setting import VAE_PATH, UNET_CONFIG_PATH, TRAIN_OUTPUT_DIR


def train(model, vae, device, train_loader, optimizer, epoch):
    iters = 0
    for epoch in tqdm(range(epoch)):
        for batch_idx, (target_image, previous_image, masked_image, audio_feature) in tqdm(enumerate(train_loader)):
            target_image, previous_image, masked_image, audio_feature = (
                target_image.to(device),
                previous_image.to(device),
                masked_image.to(device),
                audio_feature.to(device)
            )
            # 获取目标的latents
            latents = vae.encode(target_image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            # 获取输入的latents
            masked_latents = vae.encode(masked_image).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            # 获取邻近图像的latents
            previous_image = vae.encode(previous_image).latent_dist.sample()
            previous_image = previous_image * vae.config.scaling_factor
            # 拼接输入
            latent_model_input = torch.cat([masked_latents, previous_image], dim=1)

            image_pred = model(latent_model_input, 0, encoder_hidden_states=audio_feature.to(device)).sample
            loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 50 == 0:
                print(f"epoch: {epoch + 1}, iters: {batch_idx}, loss: {loss.item()}")
            iters += 1
            if (iters + 1) % 100 == 0:
                torch.save(model.state_dict(), TRAIN_OUTPUT_DIR / f'musetalk--iters--{iters + 1}.pt')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
    # 加载模型
    with open(UNET_CONFIG_PATH, "r") as f:
        unet_config = json.load(f)
    unet = UNet2DConditionModel(**unet_config).to(device)
    # 加载数据
    train_dataset = MuseTalkDataset()
    train_loader = DataLoader(train_dataset, batch_size=8)
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    # 创建优化器
    optimizer = optim.AdamW(
        unet.parameters(),
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    train(unet, vae, device, train_loader, optimizer, epoch=10)


if __name__ == '__main__':
    main()
