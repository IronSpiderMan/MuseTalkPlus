import sys
import itertools
from pathlib import Path
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration

sys.path.append('.')
from common.setting import settings
from musetalk.utils import save_model
from musetalk.datasets import MuseTalkDataset
from musetalk.models.musetalk import MuseTalkModel, PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_step = 0
checkpoint_infos = {"minimal_loss": 100, "iters": 0, "checkpoints": []}
minial_loss = 100
Path(settings.train.output).mkdir(parents=True, exist_ok=True)


def training_loop(
        models, train_loader, test_loader, optimizer, epochs, accelerator, gamma=2.0,
        max_grad_norm=1, output_dir="outputs"
):
    global global_step
    train_losses, lip_losses, latent_losses = [], [], []
    model, vae, pe = models
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=100000 * accelerator.num_processes,
    )
    model, optimizer, lr_scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, test_loader
    )

    # 训练
    for epoch in range(epochs):
        model.train()
        for step, (target_image, reference_image, masked_image, audio_feature) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            global_step += 1
            # target_image的形状为batch_size *  3 * img_size * img_size
            # audio_feature的形状为batch_size * h_dim * e_dim
            with accelerator.accumulate(model):
                # 如果使用了syncnet，则维度数为5，将dim1转换成batch_size
                if target_image.ndim == 5:
                    target_image = target_image.view(-1, 3, 256, 256)
                    reference_image = reference_image.view(-1, 3, 256, 256)
                    masked_image = masked_image.view(-1, 3, 256, 256)
                    audio_feature = audio_feature.view(-1, 50, 384)
                audio_feature = pe(audio_feature)
                # 获取目标的latents
                target_latents = vae.encode(target_image.to(dtype=vae.dtype)).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor
                # 获取输入的latents
                reference_latents = vae.encode(reference_image.to(dtype=vae.dtype)).latent_dist.sample()
                reference_latents = reference_latents * vae.config.scaling_factor
                masked_latents = vae.encode(masked_image.to(dtype=vae.dtype)).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                input_latents = torch.cat([masked_latents, reference_latents], dim=1)
                # Forward
                pred_latents = model((input_latents.float(), audio_feature))
                latent_loss = F.l1_loss(pred_latents.float(), target_latents.float(), reduction="mean")
                # 对预测图像解码
                pred_latents = (1 / vae.config.scaling_factor) * pred_latents
                pred_images = vae.decode(pred_latents.to(dtype=vae.dtype)).sample
                lip_loss = F.l1_loss(
                    pred_images[:, :, pred_images.shape[2] // 2:, :].float(),
                    target_image[:, :, target_image.shape[2] // 2:, :].float(),
                )
                loss = gamma * lip_loss + latent_loss
                train_losses.append(loss.item())
                lip_losses.append(lip_loss.item())
                latent_losses.append(latent_loss.item())
                # Backward
                accelerator.backward(loss)
                # 梯度裁剪
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(model.unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 保存 checkpoint
                if global_step % (20 * accelerator.gradient_accumulation_steps) == 0:
                    accelerator.wait_for_everyone()
                    # 只有主进程执行保存操作
                    if accelerator.is_main_process:
                        train_loss = np.mean(train_losses)
                        lip_loss = np.mean(lip_losses)
                        latent_loss = np.mean(latent_losses)
                        save_model(
                            checkpoint_infos, accelerator, model.unet, output_dir,
                            {
                                'loss': train_loss,
                                'iters': global_step,
                                'epoch': epoch + 1
                            }
                        )
                        print((
                            f"\niters: {global_step}, train_loss: {train_loss}, "
                            f"latents_loss: {latent_loss}, lip_loss: {lip_loss}"
                        ))
                        train_losses, lip_losses, latent_losses = [], [], []

                if global_step % (100 * accelerator.gradient_accumulation_steps) == 0:
                    evaluate((model, vae, pe), test_loader)


def evaluate(models, val_loader, gamma=2.0, output_dir="outputs"):
    model, vae, pe = models
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for step, (target_image, reference_image, masked_image, audio_feature) in tqdm(
                enumerate(val_loader),
                total=len(val_loader)
        ):
            if target_image.ndim == 5:
                target_image = target_image.view(-1, 3, 256, 256)
                reference_image = reference_image.view(-1, 3, 256, 256)
                masked_image = masked_image.view(-1, 3, 256, 256)
                audio_feature = audio_feature.view(-1, 50, 384)
            audio_feature = pe(audio_feature)
            # 获取目标的latents
            target_latents = vae.encode(target_image.to(dtype=vae.dtype)).latent_dist.sample()
            target_latents = target_latents * vae.config.scaling_factor
            # 获取输入的latents
            reference_latents = vae.encode(reference_image.to(dtype=vae.dtype)).latent_dist.sample()
            reference_latents = reference_latents * vae.config.scaling_factor
            masked_latents = vae.encode(masked_image.to(dtype=vae.dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            input_latents = torch.cat([masked_latents, reference_latents], dim=1)
            # Forward
            pred_latents = model((input_latents.float(), audio_feature))
            latent_loss = F.l1_loss(pred_latents.float(), target_latents.float(), reduction="mean")
            # 对预测图像解码
            pred_latents = (1 / vae.config.scaling_factor) * pred_latents
            pred_images = vae.decode(pred_latents.to(dtype=vae.dtype)).sample
            lip_loss = F.l1_loss(
                pred_images[:, :, pred_images.shape[2] // 2:, :].float(),
                target_image[:, :, target_image.shape[2] // 2:, :].float(),
            )
            loss = gamma * lip_loss + latent_loss
            total_loss += loss.item() / len(val_loader)

            if step >= (len(val_loader) - 1):
                outputs = torch.concat([pred_images, target_image], dim=0)
                outputs = make_grid(outputs.cpu(), outputs.shape[0] // 2, padding=2).permute(1, 2, 0).numpy()
                outputs = ((outputs * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255).astype(np.uint8)
                save_path = Path(output_dir) / f'{global_step}.jpg'
                cv2.imwrite(str(save_path), outputs)
                print((
                    f"\niters: {global_step}, train_loss: {total_loss}, "
                    f"latents_loss: {latent_loss.item()}, lip_loss: {lip_loss.item()}"
                ))
    return total_loss


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
        "--learning_rate",
        type=int,
        default=5e-5
    )
    parser.add_argument(
        "--use_fp16",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--audio_window",
        type=int,
        default=settings.train.audio_window
    )
    parser.add_argument(
        "--reference_window",
        type=int,
        default=settings.train.reference_window
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
    if args.use_fp16:
        weight_dtype_frozen = torch.float16
    else:
        weight_dtype_frozen = torch.float32
    model = MuseTalkModel(settings.models.unet_path).to(device)
    train_ds = MuseTalkDataset(audio_window=args.audio_window, reference_window=args.reference_window, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    test_ds = MuseTalkDataset(audio_window=args.audio_window, reference_window=args.reference_window, split="test")
    subset_indices = torch.randperm(len(test_ds))[:500]
    test_loader = DataLoader(
        Subset(test_ds, subset_indices),
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    vae = AutoencoderKL.from_pretrained(settings.models.vae_path, subfolder="vae").to(device, dtype=weight_dtype_frozen)
    vae.requires_grad_(False)
    pe = PositionalEncoding().to(device)
    pe.requires_grad_(False)

    training_loop(
        (model, vae, pe), train_loader, test_loader, optimizer, args.epochs, accelerator=accelerator,
        gamma=args.gamma, output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
