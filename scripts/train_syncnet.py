import sys
import itertools
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration

sys.path.append('.')
from common.setting import settings
from musetalk.datasets import SyncNetDataset
from musetalk.models.sync_net import SyncNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(__name__)

TRAIN_OUTPUT_DIR = Path(settings.train.output)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 1


def training_loop(
        model, train_loader, test_loader, optimizer, loss_fn, epochs, accelerator,
        max_grad_norm=1, output_dir="syncnet"
):
    global global_step
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=10000 * accelerator.num_processes,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )
    model.train()
    for epoch in range(epochs):
        for step, (images, audios, labels) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            with accelerator.accumulate(model):
                # Forward
                image_embeddings, audio_embeddings = model((images, audios))
                d = F.cosine_similarity(image_embeddings, audio_embeddings)
                loss = loss_fn(d.unsqueeze(1), labels.to(dtype=image_embeddings.dtype))
                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 梯度裁剪
                    params_to_clip = itertools.chain(model.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    # 累计梯度后更新参数
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # 保存checkpoint
                if global_step % 500 == 0:
                    if accelerator.is_main_process:
                        print(f"\niters: {global_step}, loss: {loss.item()}")
                        save_path = Path(output_dir) / f"checkpoint-{global_step}.pt"
                        accelerator.save(accelerator.get_state_dict(model), save_path)
                        logger.info(f"Saved state to {save_path}")
                global_step += 1
        # 在每个 epoch 结束后保存一次
        if accelerator.is_main_process:
            save_path = Path(output_dir) / f"checkpoint-epoch-{epoch}.pt"
            accelerator.save(accelerator.get_state_dict(model), save_path)
            logger.info(f"Saved state to {save_path} at end of epoch {epoch}")
        # 评估模型
        val_loss = evaluate(model, test_loader, loss_fn)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")
        logger.info(f"Epoch {epoch}, Validation Loss: {val_loss}")


def evaluate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, audios, labels in val_loader:
            image_embeddings, audio_embeddings = model((images, audios))
            d = F.cosine_similarity(image_embeddings, audio_embeddings)
            loss = loss_fn(d.unsqueeze(1), labels.to(dtype=image_embeddings.dtype))
            total_loss += loss.item()
            num_batches += 1

    model.train()  # 恢复训练模式
    return total_loss / num_batches if num_batches > 0 else float('inf')


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
        "--audio_window",
        type=int,
        default=settings.train.audio_window
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
        default="outputs/syncnet"
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
    model = SyncNet().to(device)

    train_loader = DataLoader(
        SyncNetDataset(audio_window=args.audio_window, split="train"),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    test_loader = DataLoader(
        SyncNetDataset(audio_window=args.audio_window, split="test"),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    training_loop(
        model, train_loader, test_loader, optimizer, nn.BCELoss(), args.epochs, accelerator=accelerator,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
