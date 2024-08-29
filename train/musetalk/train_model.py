import os
import sys
import itertools
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL

sys.path.append('.')
from common.setting import settings
from train.musetalk.datasets import MuseTalkDataset
from musetalk.models import MuseTalkModel, PositionalEncoding

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger(__name__)

TRAIN_OUTPUT_DIR = Path(settings.train.output)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0


def training_loop(
        epochs, lr, batch_size, accelerator, audio_window=0, related_window=5, gamma=2.0,
        max_grad_norm=1, output_dir="outputs"
):
    train_loader = DataLoader(
        MuseTalkDataset(audio_window=audio_window, related_window=related_window), batch_size=batch_size, num_workers=4,
        pin_memory=False,
    )
    # 加载需要的模型
    vae = AutoencoderKL.from_pretrained(settings.models.vae_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    pe = PositionalEncoding()
    pe.requires_grad_(False)
    model = MuseTalkModel(settings.models.unet_path).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0 * accelerator.num_processes,
        num_training_steps=100000 * accelerator.num_processes,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )

    # 训练
    for epoch in range(epochs):
        model.train()
        for step, (target_image, avatar_image, masked_image, audio_feature) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            with accelerator.accumulate(model):
                vae = vae.half()
                audio_feature = pe(audio_feature)
                # 获取目标的latents
                target_latents = vae.encode(target_image.to(dtype=vae.dtype)).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor
                # 获取输入的latents
                avatar_latents = vae.encode(avatar_image.to(dtype=vae.dtype)).latent_dist.sample()
                avatar_latents = avatar_latents * vae.config.scaling_factor
                masked_latents = vae.encode(masked_image.to(dtype=vae.dtype)).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                input_latents = torch.cat([masked_latents, avatar_latents], dim=1)
                # Forward
                pred_latents = model((input_latents, audio_feature))
                loss_latents = F.l1_loss(pred_latents.float(), target_latents.float(), reduction="mean")
                # 对预测图像解码
                pred_latents = (1 / vae.config.scaling_factor) * pred_latents
                pred_images = vae.decode(pred_latents).sample
                loss_lip = F.l1_loss(
                    pred_images[:, :, pred_images.shape[2] // 2:, :].float(),
                    target_image[:, :, :target_image.shape[2] // 2].float(),
                )
                loss = gamma * loss_lip + loss_latents

                # Backward
                accelerator.backward(loss)
                # 梯度裁剪
                if accelerator.sync_gradients:
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(model.unet.parameters())
                        )
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 保存checkpoint
                if accelerator.sync_gradients:
                    global global_step
                    global_step += 1
                    if global_step % 1000 == 0:
                        print(f"iters: {global_step}, loss: {loss.item()}")
                        if accelerator.is_main_process:
                            save_path = Path(output_dir) / f"checkpoint-{global_step}"
                            accelerator.save(accelerator.get_state_dict(model.unet), save_path)
                            logger.info(f"Saved state to {save_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=settings.train.batch_size
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.train.epochs
    )
    parser.add_argument(
        "--audio_window",
        type=int,
        default=settings.train.audio_window
    )
    parser.add_argument(
        "--related_window",
        type=int,
        default=settings.train.related_window
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=settings.train.gamma
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1
    )
    parser.add_argument(
        "--save_limit",
        type=int,
        default=10
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.save_limit, project_dir=args.output_dir, logging_dir=str(logging_dir)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=project_config,
    )

    training_loop(
        args.epochs, lr=1e-5, batch_size=args.batch_size, accelerator=accelerator, audio_window=args.audio_window,
        related_window=args.related_window, gamma=args.gamma, output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
