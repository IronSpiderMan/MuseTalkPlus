import os, random, cv2

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from setting import AUDIO_FEATURE_DIR, VIDEO_FRAME_DIR

syncnet_T = 1
RESIZED_IMG = 256

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15), (15, 16),  # 下颌线
    (17, 18), (18, 19), (19, 20), (20, 21),  # 左眉毛
    (22, 23), (23, 24), (24, 25), (25, 26),  # 右眉毛
    (27, 28), (28, 29), (29, 30),  # 鼻梁
    (31, 32), (32, 33), (33, 34), (34, 35),  # 鼻子
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # 左眼
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # 右眼
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),  # 上嘴唇 外延
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # 下嘴唇 外延
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)  # 嘴唇内圈
]


class MuseTalkDataset(Dataset):
    def __init__(
            self,
            use_audio_length_left=1,
            use_audio_length_right=1,
            whisper_model_type="tiny"
    ):
        self.all_data = {}
        self.audio_feature = [use_audio_length_left, use_audio_length_right]
        self.whisper_model_type = whisper_model_type
        self.use_audio_length_left = use_audio_length_left
        self.use_audio_length_right = use_audio_length_right
        self.audio_window = 1

        if self.whisper_model_type == "tiny":
            self.whisper_path = './models/whisper'
            self.whisper_feature_W = 5
            self.whisper_feature_H = 384
        elif self.whisper_model_type == "largeV2":
            self.whisper_path = '...'
            self.whisper_feature_W = 33
            self.whisper_feature_H = 1280
        self.whisper_feature_concateW = self.whisper_feature_W * 2 * (
                self.use_audio_length_left + self.use_audio_length_right + 1)  # 5*2*（2+2+1）= 50
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

    def load_frame_with_window(self, video_name, frame_idx: int):
        file_lists = [self.all_data[video_name]['image_files'][idx] for idx in
                      range(frame_idx, frame_idx + syncnet_T)]
        results = np.zeros((len(file_lists), RESIZED_IMG, RESIZED_IMG, 3), dtype=np.uint8)
        for idx, file in enumerate(file_lists):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (RESIZED_IMG, RESIZED_IMG))
            results[idx, ::] = img
        return torch.tensor(np.transpose(results / 255., (0, 3, 1, 2)))

    def load_audio_feature_with_window(self, video_name, frame_idx: int):
        if frame_idx == 0 or frame_idx == len(self.all_data[video_name]['audio_files']) - 1:
            return None
        file_lists = [self.all_data[video_name]['audio_files'][idx] for idx in
                      range(frame_idx - self.audio_window, frame_idx + self.audio_window + 1)]
        results = np.zeros((len(file_lists), 50, 384))
        for idx, file in enumerate(file_lists):
            results[idx, ::] = np.load(file)
        return torch.squeeze(torch.FloatTensor(results.reshape(1, -1, 384)))

    @staticmethod
    def filename2num(filepath):
        return int(os.path.basename(filepath).split(".")[0])

    def __len__(self):
        return min([len(self.all_data[video_name]['image_files']) for video_name in self.all_data.keys()])
        # return len(self.all_data)

    def __getitem__(self, idx):
        # 随机选一个视频
        video_name = random.choice(list(self.all_data.keys()))
        video_data = self.all_data[video_name]
        # 选一张图片
        image_file = random.choice(video_data['image_files'])
        image_idx = int(os.path.basename(image_file).split(".")[0])
        # 选一张邻近图片
        ref_image_idx = random.randint(
            max(0, image_idx - 5),
            min(len(video_data['image_files']) - 1, image_idx + 5)
        )
        target_image = self.load_frame_with_window(video_name, image_idx)
        ref_image = self.load_frame_with_window(video_name, ref_image_idx)
        # 创建mask
        mask = torch.zeros((ref_image.shape[2], ref_image.shape[3]))
        mask[:ref_image.shape[2] // 2, :] = 1
        # 对图片mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # 创建遮罩后的图像
        masked_image = target_image * (mask > 0.5)
        # 获取对应音频即window中的音频
        audio_feature = self.load_audio_feature_with_window(video_name, image_idx)
        # print(f"{'*' * 10} 各个数据的形状 {'*' * 10}")
        # print("target_image: ", target_image.shape)
        # print("ref_image: ", ref_image.shape)
        # print("masked_image: ", masked_image.shape)
        # print("mask: ", mask.shape)
        # print("audio_feature: ", audio_feature.shape)
        return target_image[0], ref_image[0], masked_image[0], mask, audio_feature


if __name__ == "__main__":
    val_data = MuseTalkDataset(
        use_audio_length_left=2,
        use_audio_length_right=2,
        whisper_model_type="tiny"
    )
    dataloader = DataLoader(val_data, batch_size=1)
    for i in dataloader:
        break
