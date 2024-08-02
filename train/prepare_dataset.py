import cv2
import os
import glob
from os import makedirs
import sys

import numpy as np
import tqdm

sys.path.append('.')
from musetalk.whisper.audio_feature_extractor import AudioFeatureExtractor
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.utils import video2images, video2audio

afe = AudioFeatureExtractor()


def process_video(video_path="./data/video/zack.mp4"):
    video_name = os.path.basename(video_path).split('.')[0]
    makedirs(f"./train/data/images/{video_name}", exist_ok=True)
    makedirs(f"./train/data/audios/{video_name}", exist_ok=True)
    makedirs("./train/data/images", exist_ok=True)
    makedirs("./train/tmp/images", exist_ok=True)
    makedirs("./train/data/audios", exist_ok=True)
    makedirs("./train/tmp/audios", exist_ok=True)
    # 提取视频帧
    video2images(video_path, "./train/tmp/images")
    # 提取音频
    audio_path = video2audio(video_path, "./train/tmp/audios")
    # 提取特征
    feature_chunks = afe.extract_and_chunk_feature(audio_path, fps=26)
    # 截取脸部
    img_list = list(glob.glob(f"./train/tmp/images/{video_name}/*"))
    coord_list, frame_list = get_landmark_and_bbox(img_list, 5)
    for idx, (coord, frame, chunk) in enumerate(zip(coord_list, frame_list, feature_chunks)):
        try:
            x1, y1, x2, y2 = coord
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            dst = os.path.join(f"./train/data/images/{video_name}/{idx}.png")
            cv2.imwrite(dst, resized_crop_frame)
            dst = os.path.join(f'./train/data/audios/{video_name}/{idx}.npy')
            np.save(dst, chunk)
        except Exception as e:
            print(e)


def process_videos(video_dir="./train/videos"):
    for file in tqdm.tqdm(glob.glob(video_dir + "/*.mp4")):
        process_video(file)


if __name__ == '__main__':
    process_videos()
