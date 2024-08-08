import os
import sys
import glob
import shutil

sys.path.append('.')

import cv2
import tqdm
import numpy as np

from musetalk.whisper.audio_feature_extractor import AudioFeatureExtractor
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.utils import video2images, video2audio
from common.setting import VIDEO_FRAME_DIR, AUDIO_FEATURE_DIR, TMP_FRAME_DIR, TMP_AUDIO_DIR, TMP_DATASET_DIR

afe = AudioFeatureExtractor()


def process_video(video_path="./data/video/zack.mp4"):
    video_name = os.path.basename(video_path).split('.')[0]
    os.makedirs(VIDEO_FRAME_DIR / video_name, exist_ok=True)
    os.makedirs(AUDIO_FEATURE_DIR / video_name, exist_ok=True)
    os.makedirs(TMP_FRAME_DIR, exist_ok=True)
    os.makedirs(TMP_AUDIO_DIR, exist_ok=True)
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
            dst = AUDIO_FEATURE_DIR / video_name / f"{idx}.npy"
            np.save(str(dst), chunk)
        except Exception as e:
            print(e)
    shutil.rmtree(TMP_DATASET_DIR)


def process_videos(video_dir="./datasets/videos"):
    for file in tqdm.tqdm(glob.glob(video_dir + "/*.mp4")):
        process_video(file)


if __name__ == '__main__':
    process_videos()
