import subprocess

import torch


def datagen(
        whisper_chunks,
        vae_encode_latents,
        batch_size=8,
        audio_window=5,
        delay_frames=0,
):
    whisper_batch, latent_batch = [], []
    for i in range(audio_window, whisper_chunks.shape[0] - audio_window):
        idx = (i + delay_frames) % vae_encode_latents.shape[0]
        whisper_batch.append(whisper_chunks[i - audio_window: i + audio_window + 1, :, :].reshape(1, -1, 384))
        latent_batch.append(vae_encode_latents[idx:idx + 1])
        if len(latent_batch) >= batch_size:
            yield torch.cat(whisper_batch, dim=0), torch.cat(latent_batch, dim=0)
            whisper_batch, latent_batch = [], []
    if len(latent_batch) > 0:
        yield torch.cat(whisper_batch, dim=0), torch.cat(latent_batch, dim=0)


def images2video(images_dir, output, fps=25):
    subprocess.run([
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(images_dir / '%08d.jpg'),  # 匹配生成的帧
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-y',
        str(output)
    ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
