import os
import sys
import json

sys.path.append('.')

import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, UNet2DConditionModel

from train.musetalk.datasets import MuseTalkDataset
from common.setting import VAE_PATH, UNET_CONFIG_PATH, TRAIN_OUTPUT_DIR, TRAIN_OUTPUT_LOGS_DIR


def train(model, vae, device, train_loader, optimizer, epoch, accelerator, scheduler):
    iters = 0
    for epoch in range(epoch):
        for batch_idx, (target_image, previous_image, masked_image, audio_feature) in tqdm(enumerate(train_loader),
                                                                                           total=len(train_loader)):
            optimizer.zero_grad()
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
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            scheduler.step()
            if (batch_idx + 1) % 50 == 0:
                print(f"epoch: {epoch + 1}, iters: {batch_idx}, loss: {loss.item()}")
            iters += 1
            if (iters + 1) % 1000 == 0:
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
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=8)
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_LOGS_DIR, exist_ok=True)
    # 创建加速器
    project_config = ProjectConfiguration(
        total_limit=10,
        project_dir=TRAIN_OUTPUT_DIR,
        logging_dir=TRAIN_OUTPUT_LOGS_DIR
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp32",
        log_with="tensorboard",
        project_config=project_config
    )
    # 创建优化器
    optimizer = optim.AdamW(
        unet.parameters(),
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    # 创建scheduler
    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
    )
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    unet.train()
    vae.requires_grad_(False)
    train(unet, vae, device, train_loader, optimizer, 10, accelerator, lr_scheduler)


if __name__ == '__main__':
    main()
