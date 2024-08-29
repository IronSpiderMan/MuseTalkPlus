import subprocess

import numpy as np
import torch


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
