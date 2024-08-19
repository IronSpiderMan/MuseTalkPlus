import os
import sys
import datetime
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL

sys.path.append('.')

from musetalk_plus.train.datasets import MuseTalkDataset
from musetalk_plus.models import MuseTalkModel
from common.setting import VAE_PATH, TRAIN_OUTPUT_DIR, UNET_PATH
from musetalk.models.unet import PositionalEncoding

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
vae.requires_grad_(False)
pe = PositionalEncoding().to(device)


def training_loop(epochs, lr, batch_size, mixed_precision='no', max_checkpoints=10, audio_window=5):
    train_loader = DataLoader(
        MuseTalkDataset(audio_window=audio_window), batch_size=batch_size, num_workers=4, pin_memory=True
    )
    model = MuseTalkModel(UNET_PATH).to(device)
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

    # 初始化用于存储检查点信息的变量
    checkpoint_list = []
    min_loss = float('inf')
    min_loss_checkpoint = None

    # 训练
    for epoch in range(epochs):
        model.train()
        for step, (target_image, avatar_image, masked_image, audio_feature) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            target_image, avatar_image, masked_image, audio_feature = (
                target_image.to(device),
                avatar_image.to(device),
                masked_image.to(device),
                audio_feature.to(device)
            )
            # 获取目标的latents
            latents = vae.encode(target_image.to(vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            # 获取输入的latents
            avatar_image = vae.encode(avatar_image.to(vae.dtype)).latent_dist.sample()
            avatar_image = avatar_image * vae.config.scaling_factor
            masked_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            input_latents = torch.cat([avatar_image, masked_latents], dim=1)
            # audio_feature = pe(audio_feature)
            # Forward
            image_pred = model((input_latents, audio_feature))
            loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")
            # Backward
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 保存检查点
            if (step + 1) % 1000 == 0:
                accelerator.wait_for_everyone()
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                accelerator.print(f"epoch【{epoch}】@{now} --> loss = {loss:.5f}")

                # 保存当前检查点
                checkpoint_path = TRAIN_OUTPUT_DIR / f"checkpoint-epoch-{epoch + 1}-iters-{step + 1}-loss-{loss:.5f}.pt"
                accelerator.save(accelerator.get_state_dict(model), checkpoint_path)
                checkpoint_list.append(checkpoint_path)

                # 维护最多10个检查点
                if len(checkpoint_list) > max_checkpoints:
                    # 删除最早的检查点
                    oldest_checkpoint = checkpoint_list.pop(0)
                    if oldest_checkpoint != min_loss_checkpoint:
                        os.remove(oldest_checkpoint)

                # 更新最小损失检查点
                if loss < min_loss:
                    min_loss = loss
                    min_loss_checkpoint = checkpoint_path

                    # 复制最小损失的检查点
                    min_loss_checkpoint_copy = TRAIN_OUTPUT_DIR / "best_checkpoint.pt"
                    accelerator.save(accelerator.get_state_dict(model.unet), min_loss_checkpoint_copy)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        "--audio_window",
        type=int,
        default=5
    )
    return parser.parse_args()


def main():
    args = parse_args()
    training_loop(
        args.epochs, lr=1e-5, batch_size=args.batch_size, mixed_precision="no", audio_window=args.audio_window
    )


if __name__ == '__main__':
    main()
