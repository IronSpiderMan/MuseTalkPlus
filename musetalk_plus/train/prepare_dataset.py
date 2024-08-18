import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

import cv2
import tqdm
import torch
import numpy as np

sys.path.append('.')

from musetalk_plus.whisper.feature_extractor import AudioFrameExtractor
from musetalk_plus.faces.face_recognize import FaceRecognizer

from common.utils import recreate_multiple_dirs, read_images, video2images, video2audio
from common.setting import (
    TMP_FRAME_DIR, TMP_AUDIO_DIR, TMP_DATASET_DIR,
    VIDEO_FRAME_DIR, AUDIO_FEATURE_DIR, VIDEO_LATENT_DIR
)

device = "cuda" if torch.cuda.is_available() else "cpu"
afe = AudioFrameExtractor(r"F:\models\whisper-tiny-zh")
fr = FaceRecognizer()


def process_video(video_path="./data/video/zack.mp4", fixed_face=True):
    face_location = None
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
    feature_chunks = afe.extract_frames(audio_path)
    # 截取脸部
    path_pattern = TMP_FRAME_DIR / video_name / "*"
    img_list = list(glob.glob(str(path_pattern)))
    if face_location is None:
        y1, x2, y2, x1 = fr.face_locations(img_list[0])
        face_location = [x1, y1, x2, y2]
    coord_list = [[*face_location] for _ in range(len(img_list))]
    frame_list = read_images(img_list)
    for idx, (coord, frame, chunk) in enumerate(zip(coord_list, frame_list, feature_chunks)):
        try:
            x1, y1, x2, y2 = coord
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            dst = VIDEO_FRAME_DIR / video_name / f"{idx}.png"
            cv2.imwrite(str(dst), resized_crop_frame)
            dst = AUDIO_FEATURE_DIR / video_name / f"{idx}.npy"
            np.save(str(dst), chunk)
        except Exception as e:
            print(e)
    shutil.rmtree(TMP_DATASET_DIR)


def process_videos(video_dir="./datasets/videos", include_latents=False):
    # video_list = Path(video_dir).glob("/*.mp4")
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
    args = parse_args()
    process_videos(video_dir=args.videos_dir, include_latents=args.include_latents)


if __name__ == '__main__':
    main()
