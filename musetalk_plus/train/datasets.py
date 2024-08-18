import os
import sys
import random

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

sys.path.append('.')

from common.setting import AUDIO_FEATURE_DIR, VIDEO_FRAME_DIR
from musetalk_plus.processors import ImageProcessor

RESIZED_IMG = 256
device = "cuda" if torch.cuda.is_available() else "cpu"


class MuseTalkDataset(Dataset):
    def __init__(
            self,
            audio_window=5,
    ):
        self.all_data = {}
        self.audio_window = audio_window

        self.whisper_feature_W = 2
        self.whisper_feature_H = 384
        self.image_processor = ImageProcessor()
        self.load_filenames()

    @staticmethod
    def sort_files(files):
        return sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def load_filenames(self):
        for video_name in os.listdir(VIDEO_FRAME_DIR):
            self.all_data[video_name] = {
                "image_files": [],
                "audio_files": []
            }
            # 各个视频对应的图片路径
            images_dir = os.path.join(VIDEO_FRAME_DIR, video_name)
            for filename in self.sort_files(os.listdir(images_dir)):
                self.all_data[video_name]["image_files"].append(
                    os.path.join(images_dir, filename)
                )
            # 各个视频对应的音频路径
            audios_dir = os.path.join(AUDIO_FEATURE_DIR, video_name)
            for filename in self.sort_files(os.listdir(audios_dir)):
                self.all_data[video_name]["audio_files"].append(
                    os.path.join(audios_dir, filename)
                )
            # 保证图片和音频是帧数一样
            max_length = min(
                len(self.all_data[video_name]['image_files']),
                len(self.all_data[video_name]['audio_files']),
            )
            self.all_data[video_name]['image_files'] = self.all_data[video_name]['image_files'][:max_length]
            self.all_data[video_name]['audio_files'] = self.all_data[video_name]['audio_files'][:max_length]
            # 过滤5秒以下的视频
            if len(self.all_data[video_name]['image_files']) < 25 * 5:
                del self.all_data[video_name]
        return self.all_data

    def load_audio_feature_with_window(self, video_name, frame_idx: int):
        if frame_idx - self.audio_window < 0 or frame_idx + self.audio_window == len(
                self.all_data[video_name]['audio_files']) - 1:
            file_list = [self.all_data[video_name]['audio_files'][frame_idx]] * (self.audio_window * 2 + 1)
        else:
            file_list = [self.all_data[video_name]['audio_files'][idx] for idx in
                         range(frame_idx - self.audio_window, frame_idx + self.audio_window + 1)]
        results = np.zeros((len(file_list), self.whisper_feature_W, self.whisper_feature_H))
        for idx, file in enumerate(file_list):
            results[idx, ::] = np.load(file)
        return torch.FloatTensor(results.reshape(-1, self.whisper_feature_H))

    def load_frames(self, video_name, frame_idx: int):
        frame_list = [frame_idx, 0]
        images = []
        for frame_idx in frame_list:
            images.append(self.load_frame(video_name, frame_idx))
        images.append(self.load_frame(video_name, frame_idx, True))
        # images三个元素分别为, target_image, avatar_image, masked_image
        return images

    def load_frame(self, video_name, frame_idx, half_masked=False):
        image = cv2.imread(self.all_data[video_name]['image_files'][frame_idx])
        return self.image_processor(image, half_mask=half_masked)

    @staticmethod
    def filename2num(filepath):
        return int(os.path.basename(filepath).split(".")[0])

    def __len__(self):
        return sum([len(self.all_data[video_name]['image_files']) for video_name in self.all_data.keys()])

    def __getitem__(self, idx):
        # 随机选一个视频
        video_name = random.choice(list(self.all_data.keys()))
        video_data = self.all_data[video_name]
        # TODO masked_image应该使用非target_image旁边的图像，因为在推理时我们不知道target_image
        # 在audio_window ~ len(images)-audio_window范围选一张图片
        frame_idx = random.randint(self.audio_window, len(video_data['image_files']) - self.audio_window - 1)
        target_image, avatar_image, masked_image = self.load_frames(video_name, frame_idx)
        # 获取对应音频即window中的音频
        audio_feature = self.load_audio_feature_with_window(video_name, frame_idx)
        return target_image, avatar_image, masked_image, audio_feature


if __name__ == "__main__":
    # get_face_mask(cv2.imread("./results/tjl/full_images/00000000.png"))
    val_data = MuseTalkDataset()
    dataloader = DataLoader(val_data, batch_size=1)
    for i in dataloader:
        # print(i)
        ti, ri, mi, af = i
        # print(ti.shape)
        # print(mi.shape)
        # print(af.shape)
        break