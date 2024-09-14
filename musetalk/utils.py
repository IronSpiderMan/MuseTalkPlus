import json
import shutil
import subprocess
from pathlib import Path

import torch
import numpy as np
from torch import nn
from accelerate import Accelerator


def datagen(
        whisper_chunks,
        vae_encode_latents,
        batch_size=8,
        delay_frames=0,
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i + delay_frames) % vae_encode_latents.shape[0]
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            yield torch.Tensor(np.array(whisper_batch)), torch.Tensor(np.array(latent_batch))
            whisper_batch, latent_batch = [], []
    if len(latent_batch) > 0:
        yield torch.Tensor(np.array(whisper_batch)), torch.Tensor(np.array(latent_batch))


def images2video(images_dir, output, fps=25):
    subprocess.run([
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(images_dir / '%08d.jpg'),  # 匹配生成的帧
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-y',
        str(output)
    ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def merge_audio_video(video_path, audio_path, output_path):
    # 调用 ffmpeg 命令合并音频和视频
    subprocess.run([
        'ffmpeg',
        '-i', str(video_path),  # 输入视频文件
        '-i', str(audio_path),  # 输入音频文件
        '-c:v', 'copy',  # 复制视频流
        '-c:a', 'aac',  # 使用 AAC 编码音频
        '-strict', 'experimental',  # 允许使用实验性的 AAC 编码
        '-map', '0:v:0',  # 选择第一个视频流
        '-map', '1:a:0',  # 选择第一个音频流
        '-shortest',  # 确保输出文件的时长与较短的输入文件相同
        '-y',  # 覆盖输出文件
        str(output_path)  # 输出文件路径
    ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)


def save_model(
        checkpoint_infos: dict, accelerator: Accelerator, model: nn.Module,
        output_dir: str, train_infos: dict, total_limit=10
):
    """
    checkpoint_infos: {
        "minimal_loss": 0.0,
        "checkpoints": [
            {'loss': train_loss, 'iters': iters, 'epoch': epoch + 1, 'filepath', 'path/to/model'},
            ....
        ],
        "iters": 1000
    }
    """
    checkpoints = checkpoint_infos['checkpoints']
    if len(checkpoints) >= total_limit:
        # 删除最早的模型
        try:
            Path(checkpoints[0]['filepath']).unlink()
            del checkpoints[0]
            # 重命名其余模型
            for idx, checkpoint in enumerate(checkpoints):
                new_path = Path(output_dir) / f'pytorch_model_{idx}.bin'
                Path(checkpoint['filepath']).rename(new_path)
                checkpoints[idx]['filepath'] = str(new_path)
            checkpoint_infos['checkpoints'] = checkpoints
        except FileNotFoundError:
            pass

    # 保存模型
    save_path = Path(output_dir) / f'pytorch_model_{len(checkpoints)}.bin'
    accelerator.save(accelerator.get_state_dict(model), save_path)
    train_infos['filepath'] = str(save_path)
    checkpoints.append(train_infos)

    # 更新最优模型
    if train_infos['loss'] < checkpoint_infos['minimal_loss']:
        checkpoint_infos['minimal_loss'] = train_infos['loss']
        shutil.copy(save_path, Path(output_dir) / 'best_model.bin')
    checkpoint_infos['checkpoints'] = checkpoints
    checkpoint_infos['iters'] = train_infos['iters']
    with open(Path(output_dir) / 'checkpoints.json', 'w', encoding='utf-8') as f:
        json.dump(checkpoint_infos, f, indent=4, ensure_ascii=False)
    return checkpoint_infos
