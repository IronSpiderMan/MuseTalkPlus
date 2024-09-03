import json
import random
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from common.setting import settings
from common.utils import read_images, video2images, video2audio, recreate_multiple_dirs

from musetalk.faces.face_analysis import FaceAnalyst
from musetalk.audio.audio_feature_extract import AudioFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
afe: AudioFeatureExtractor
fa = FaceAnalyst(settings.models.dwpose_config_path, settings.models.dwpose_model_path)

IMAGE_SIZE = settings.common.image_size


def process_video(video_path, face_shift=None):
    """
    video_path: 视频路径
    face_shift: 人脸往上偏移的像素值，如果为None则偏移到额头

    return {
        "image_files": [path/to/image01, path/to/image02],
        "audio_files": [path/to/audio01, path/to/audio02],
    }
    """
    video_data = {
        "image_files": [],
        "audio_files": [],
    }
    # 临时目录
    tmp_frame_dir = Path(settings.dataset.base_dir) / 'tmp' / 'images'
    tmp_audio_dir = Path(settings.dataset.base_dir) / 'tmp' / 'audios'
    recreate_multiple_dirs([tmp_frame_dir, tmp_audio_dir])

    video_name = video_path.stem
    # 视频部分的预处理
    video_frame_base_dir = Path(settings.dataset.images_dir)
    video_frame_dir = video_frame_base_dir / video_name
    # 目录不存在或目录为空才创建
    if not video_frame_dir.exists() or not any(video_frame_dir.iterdir()):
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        video2images(video_path, tmp_frame_dir)
        frame_list = read_images([str(img) for img in tmp_frame_dir.glob('*')], to_rgb=False)
        for fidx, frame in tqdm(
                enumerate(frame_list),
                total=len(frame_list),
                desc=f"Processing video: {video_name}"
        ):
            pts = fa.analysis(frame)
            bbox = fa.face_location(pts, shift=face_shift)
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
            dst = video_frame_dir / f"{fidx:08d}.png"
            cv2.imwrite(str(dst), resized_crop_frame)
            video_data['image_files'].append(str(dst))
        if not any(video_frame_dir.iterdir()):
            shutil.rmtree(video_frame_dir)
            return
    else:
        video_data['image_files'] = [str(i) for i in video_frame_dir.glob('*')]
        print(f"Video {video_name} is already processed")

    # 音频部分的预处理
    audio_feature_base_dir = Path(settings.dataset.audios_dir)
    audio_feature_dir = audio_feature_base_dir / video_name
    if not audio_feature_dir.exists() or not any(audio_feature_dir.iterdir()):
        audio_feature_dir.mkdir(parents=True, exist_ok=True)
        audio_path = video2audio(video_path, tmp_audio_dir)
        feature_chunks = afe.extract_features(audio_path, 0)
        for fidx, chunk in tqdm(
                enumerate(feature_chunks),
                total=len(feature_chunks),
                desc=f"Processing video {video_name} 's audio"
        ):
            dst = audio_feature_dir / f"{fidx:08d}.npy"
            np.save(str(dst), chunk)
            video_data['audio_files'].append(str(dst))
    else:
        video_data['audio_files'] = [str(i) for i in audio_feature_dir.glob('*')]
        print(f"Video {video_name}'s audio has already been processed.")
    shutil.rmtree(Path(settings.dataset.base_dir) / 'tmp')
    return video_data


def process_videos(video_dir="./datasets/videos", face_shift=None, test_split=0.2):
    video_list = list(Path(video_dir).glob("*.mp4"))
    all_data = {}
    for video_path in tqdm(video_list, total=len(video_list), desc='Processing videos'):
        if video_data := process_video(video_path, face_shift):
            all_data[video_path.stem] = video_data

    keys = list(all_data.keys())
    train_count = int(len(keys) * (1 - test_split))
    train_keys = random.sample(keys, train_count)
    test_keys = list(set(keys).difference(set(train_keys)))
    train_data = {key: all_data[key] for key in train_keys}
    test_data = {key: all_data[key] for key in test_keys}
    with open(Path(settings.dataset.base_dir) / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(Path(settings.dataset.base_dir) / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


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
    args = parse_args()
    global afe
    afe = AudioFeatureExtractor(settings.models.whisper_path, device=device, dtype=torch.float32)
    process_videos(video_dir=args.videos_dir, face_shift=args.face_shift, test_split=args.test_split)


if __name__ == '__main__':
    main()
