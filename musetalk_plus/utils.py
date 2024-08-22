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
