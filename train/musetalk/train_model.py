import sys
import json
import datetime

import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel

sys.path.append('.')

from train.musetalk.datasets import MuseTalkDataset
from common.setting import VAE_PATH, UNET_CONFIG_PATH, TRAIN_OUTPUT_DIR
from musetalk.models.unet import PositionalEncoding

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
vae.requires_grad_(False)
pe = PositionalEncoding().to(device)


def training_loop(epochs, lr, batch_size, mixed_precision='no'):
    train_loader = DataLoader(MuseTalkDataset(), batch_size=batch_size)
    with open(UNET_CONFIG_PATH, "r") as f:
        unet_config = json.load(f)
    model = UNet2DConditionModel(**unet_config).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=25 * lr,
        epochs=epochs, steps_per_epoch=len(train_loader)
    )
    set_seed(42)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )
    # 训练
    for epoch in range(epochs):
        model.train()
        for step, (target_image, related_image, masked_image, audio_feature) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            target_image, related_image, masked_image, audio_feature = (
                target_image.to(device),
                related_image.to(device),
                masked_image.to(device),
                audio_feature.to(device)
            )
            # 获取目标的latents
            latents = vae.encode(target_image.to(vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # 获取输入的latents
            masked_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            related_image = vae.encode(related_image.to(vae.dtype)).latent_dist.sample()
            related_image = related_image * vae.config.scaling_factor
            input_latents = torch.cat([masked_latents, related_image], dim=1)
            with torch.no_grad():
                audio_feature = pe(audio_feature)
            # Forward
            image_pred = model(input_latents, 0, encoder_hidden_states=audio_feature).sample
            loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")
            # Backward
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if (step + 1) % 1000 == 0:
                accelerator.wait_for_everyone()
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                accelerator.print(f"epoch【{epoch}】@{now} --> loss = {loss:.5f}%")
                net_dict = accelerator.get_state_dict(model)
                accelerator.save(
                    net_dict,
                    TRAIN_OUTPUT_DIR / f"checkpoint-epoch-{epoch + 1}-iters-{step + 1}-loss-{loss:.5f}.pt"
                )


def main():
    training_loop(10, lr=1e-5, batch_size=8, mixed_precision="fp16")


if __name__ == '__main__':
    main()
