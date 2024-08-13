import os
import sys
import glob
import shutil
import argparse

import cv2
import tqdm
import torch
import numpy as np

sys.path.append('.')

from musetalk.models.vae import VAE
from musetalk.utils.utils import video2images, video2audio
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.whisper.audio_feature_extractor import AudioFeatureExtractor

from common.utils import recreate_multiple_dirs
from common.setting import (
    TMP_FRAME_DIR, TMP_AUDIO_DIR, TMP_DATASET_DIR,
    VIDEO_FRAME_DIR, AUDIO_FEATURE_DIR, VIDEO_LATENT_DIR, VAE_PATH
)

afe = AudioFeatureExtractor()
vae: VAE
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_video(video_path="./data/video/zack.mp4", include_latents=False):
    video_name = os.path.basename(video_path).split('.')[0]
    recreate_multiple_dirs([
        VIDEO_FRAME_DIR / video_name,
        VIDEO_LATENT_DIR / video_name,
        AUDIO_FEATURE_DIR / video_name,
        TMP_FRAME_DIR,
        TMP_AUDIO_DIR
    ])
    # 提取视频帧
    video2images(video_path, TMP_FRAME_DIR)
    # 提取音频
    audio_path = video2audio(video_path, TMP_AUDIO_DIR)
    # 提取特征
    feature_chunks = afe.extract_and_chunk_feature(audio_path, fps=25)
    # 截取脸部
    path_pattern = TMP_FRAME_DIR / video_name / "*"
    img_list = list(glob.glob(str(path_pattern)))
    coord_list, frame_list = get_landmark_and_bbox(img_list, 5)
    for idx, (coord, frame, chunk) in enumerate(zip(coord_list, frame_list, feature_chunks)):
        try:
            x1, y1, x2, y2 = coord
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            dst = VIDEO_FRAME_DIR / video_name / f"{idx}.png"
            cv2.imwrite(str(dst), resized_crop_frame)
            if include_latents:
                dst = VIDEO_LATENT_DIR / video_name / f"{idx}.npy"
                latent = vae.get_latents_for_unet(resized_crop_frame).cpu().numpy()[0]
                np.save(str(dst), latent)
            dst = AUDIO_FEATURE_DIR / video_name / f"{idx}.npy"
            np.save(str(dst), chunk)
        except Exception as e:
            print(e)
    shutil.rmtree(TMP_DATASET_DIR)


def process_videos(video_dir="./datasets/videos", include_latents=False):
    for file in tqdm.tqdm(glob.glob(video_dir + "/*.mp4")):
        process_video(file, include_latents)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include_latents",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="./datasets/videos"
    )
    return parser.parse_args()


def main():
    global vae
    args = parse_args()
    if args.include_latents:
        vae = VAE(VAE_PATH)
        vae.vae = vae.vae.to(device)
    process_videos(video_dir=args.videos_dir, include_latents=args.include_latents)


if __name__ == '__main__':
    main()
