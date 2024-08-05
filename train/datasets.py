import os
import sys
import random

sys.path.append('.')

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from common.setting import AUDIO_FEATURE_DIR, VIDEO_FRAME_DIR

RESIZED_IMG = 256


class MuseTalkDataset(Dataset):
    def __init__(
            self,
            audio_window=1
    ):
        self.all_data = {}
        self.audio_window = audio_window

        self.whisper_feature_W = 50
        self.whisper_feature_H = 384
        self.load_filenames()

    @staticmethod
    def sort_files(files):
        return sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))

    def load_filenames(self):
        for video_name in os.listdir(VIDEO_FRAME_DIR):
            self.all_data = {video_name: {
                "image_files": [],
                "audio_files": []
            }}
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
        return self.all_data

    def load_audio_feature_with_window(self, video_name, frame_idx: int):
        if frame_idx - self.audio_window < 0 or frame_idx + self.audio_window == len(
                self.all_data[video_name]['audio_files']) - 1:
            return None
        file_lists = [self.all_data[video_name]['audio_files'][idx] for idx in
                      range(frame_idx - self.audio_window, frame_idx + self.audio_window + 1)]
        results = np.zeros((len(file_lists), self.whisper_feature_W, self.whisper_feature_H))
        for idx, file in enumerate(file_lists):
            results[idx, ::] = np.load(file)
        return torch.FloatTensor(results.reshape(-1, self.whisper_feature_H))

    @staticmethod
    def filename2num(filepath):
        return int(os.path.basename(filepath).split(".")[0])

    def __len__(self):
        return min([len(self.all_data[video_name]['image_files']) for video_name in self.all_data.keys()])

    def __getitem__(self, idx):
        # 随机选一个视频
        video_name = random.choice(list(self.all_data.keys()))
        video_data = self.all_data[video_name]
        # 选一张图片
        image_file = random.choice(video_data['image_files'])
        target_image = cv2.imread(image_file)
        target_image = cv2.resize(target_image, (RESIZED_IMG, RESIZED_IMG))
        target_image = torch.tensor(np.transpose(target_image / 255., (2, 0, 1)))
        # 创建mask
        mask = torch.zeros((target_image.shape[1], target_image.shape[2]))
        mask[:target_image.shape[1] // 2, :] = 1
        # 创建遮罩后的图像
        masked_image = target_image * mask
        # 获取对应音频即window中的音频
        audio_feature = self.load_audio_feature_with_window(video_name, self.filename2num(image_file))
        return target_image, masked_image, audio_feature


if __name__ == "__main__":
    val_data = MuseTalkDataset()
    dataloader = DataLoader(val_data, batch_size=1)
    for i in dataloader:
        ti, mi, af = i
        print(ti.shape)
        print(mi.shape)
        print(af.shape)
        break
