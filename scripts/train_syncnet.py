import os
import sys
import json
import itertools
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration

sys.path.append('.')
from common.setting import settings
from musetalk.utils import save_model
from musetalk.datasets import SyncNetDataset
from musetalk.models.sync_net import SyncNet, ContrastiveLoss

Path(settings.train.output).mkdir(parents=True, exist_ok=True)
global_step = 0
checkpoint_infos = {"minimal_loss": 100, "iters": 0, "checkpoints": []}
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def training_loop(
        model, train_loader, test_loader, optimizer, loss_fn, epochs, accelerator,
        max_grad_norm=1, output_dir="syncnet"
):
    # 存储filepath, loss, iters
    global global_step
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=10000 * accelerator.num_processes,
    )
    model, optimizer, lr_scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, test_loader
    )
    model.train()

    train_losses = []
    for epoch in range(epochs):
        for step, (images, audios, labels) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
        ):
            global_step += 1
            with accelerator.accumulate(model):
                # Forward
                image_embeddings, audio_embeddings = model((images, audios))
                loss = loss_fn(image_embeddings, audio_embeddings, labels)
                train_losses.append(loss.item())
                # Backward
                accelerator.backward(loss)
                # 梯度裁剪
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(model.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                # 更新参数
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 保存 checkpoint
                if global_step % 500 == 0:
                    accelerator.wait_for_everyone()
                    # 只有主进程执行保存操作
                    if accelerator.is_main_process:
                        train_loss = np.mean(train_losses)
                        save_model(
                            checkpoint_infos, accelerator, model, output_dir,
                            {
                                'loss': train_loss,
                                'iters': global_step,
                                'epoch': epoch + 1
                            }
                        )
                        print(f"\niters: {global_step},train_loss: {train_loss}")
                        # 评估模型
                        val_accuracy = evaluate(model, test_loader, loss_fn)
                        print(f"\nEpoch {epoch}, Validation loss: {val_accuracy}")
                        train_losses = []


def evaluate(model, val_loader, loss_fn):
    total_losses = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for idx, (images, audios, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating"):
            image_embeddings, audio_embeddings = model((images, audios))
            loss = loss_fn(image_embeddings, audio_embeddings, labels)
            total_losses += loss.item()
            total += 1

    model.train()  # Restore training mode
    average_loss = total_losses / total
    return average_loss


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
        default=10
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
    global checkpoint_infos
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=str(logging_dir)
        ),
    )
    model_path = Path(args.output_dir) / 'best_model.bin'
    if model_path.exists():
        print("Loading model from {}".format(model_path))
        model = SyncNet()
        model.load_state_dict(torch.load(model_path, map_location=accelerator.device))
        config_path = model_path.parent / 'checkpoints.json'
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                jdata = json.load(f)
                if 'checkpoints' in jdata:
                    checkpoint_infos = jdata
    else:
        model = SyncNet().to(accelerator.device)
    train_loader = DataLoader(
        SyncNetDataset(audio_window=args.audio_window, split="train"),
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    test_ds = SyncNetDataset(audio_window=args.audio_window, split="test")
    subset_indices = torch.randperm(len(test_ds))[:500]
    test_loader = DataLoader(
        Subset(test_ds, subset_indices),
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    training_loop(
        model, train_loader, test_loader, optimizer, ContrastiveLoss(), args.epochs, accelerator=accelerator,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
