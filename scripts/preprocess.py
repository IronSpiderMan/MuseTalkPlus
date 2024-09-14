import sys
import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL

sys.path.append('.')

from common.setting import settings
from musetalk.processors import ImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = settings.common.image_size
ip = ImageProcessor()
vae = AutoencoderKL.from_pretrained(settings.models.vae_path, subfolder="vae").to(device, dtype=torch.float16)
vae.requires_grad_(False)


def process_video(video_path):
    video_name = video_path.stem

    video_frame_dir = Path(settings.dataset.images_dir) / video_name
    video_latent_dir = Path(settings.dataset.latents_dir) / video_name
    audio_feature_dir = Path(settings.dataset.audios_dir) / video_name
    # 目录不存在或目录为空才创建
    if not video_latent_dir.exists() or not any(video_latent_dir.iterdir()):
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        frame_list = [str(img) for img in video_frame_dir.glob('*')]
        audio_feature_list = [np.load(str(audio)) for audio in audio_feature_dir.glob("*")]
        for fidx, (frame, audio_feature) in tqdm(
                enumerate(zip(frame_list, audio_feature_list)),
                total=len(frame_list),
                desc=f"Processing video: {video_name}"
        ):
            frame = cv2.imread(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_image = ip(frame)
            masked_image = ip(frame, half_mask=True)

            target_latent = vae.encode(target_image.to(dtype=vae.dtype)).latent_dist.sample()
            target_latent = target_latent * vae.config.scaling_factor

            masked_latent = vae.encode(masked_image.to(dtype=vae.dtype)).latent_dist.sample()
            masked_latent = masked_latent * vae.config.scaling_factor


def process_videos(video_dir="./datasets/videos", face_shift=None, test_split=0.2):
    video_list = list(Path(video_dir).glob("*.mp4"))
    for video_path in tqdm(video_list, total=len(video_list), desc='Processing videos'):
        process_video(video_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="./datasets/videos"
    )
    parser.add_argument(
        "--face_shift",
        type=Optional[int],
        default=None,
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
    )
    return parser.parse_args()


def main():
    global afe
    args = parse_args()
    process_videos(video_dir=args.videos_dir, face_shift=args.face_shift, test_split=args.test_split)


if __name__ == '__main__':
    main()
