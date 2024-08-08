import os
import sys
import random

sys.path.append('.')

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

from common.setting import AUDIO_FEATURE_DIR, VIDEO_FRAME_DIR, DWPOST_PATH
from musetalk.utils.face_detection import FaceAlignment, LandmarksType
from musetalk.utils.blending import get_image_prepare_material

RESIZED_IMG = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
model = init_model(config_file, str(DWPOST_PATH), device=device)
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def get_face_mask(image, upperbound_range=0):
    results = inference_topdown(model, image)
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    face_land_mark = keypoints[0][23:91]
    face_land_mark = face_land_mark.astype(np.int32)
    bbox = fa.get_detections_for_batch(np.asarray([image]))

    average_range_minus = []
    average_range_plus = []

    if len(bbox) == 0:
        landmark = coord_placeholder
    else:
        half_face_coord = face_land_mark[29]
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
        if upperbound_range != 0:
            half_face_coord[1] = upperbound_range + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = half_face_coord[1] - half_face_dist

        f_landmark = (
            np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]),
            np.max(face_land_mark[:, 1]))
        x1, y1, x2, y2 = f_landmark
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
            landmark = bbox[0]
        else:
            landmark = f_landmark
    mask, crop_box = get_image_prepare_material(image, landmark)
    return mask, crop_box


class MuseTalkDataset(Dataset):
    def __init__(
            self,
            audio_window=1
    ):
        self.all_data = {}
        self.audio_window = audio_window

        self.whisper_feature_W = 50
        self.whisper_feature_H = 384
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

    def load_frame_with_previous(self, video_name, frame_idx: int):
        # 读取前一张和当前图像
        file_list = [
            self.all_data[video_name]['image_files'][frame_idx - 1],
            self.all_data[video_name]['image_files'][frame_idx],
        ]
        images = []
        for file in file_list:
            image = cv2.imread(file)
            image = cv2.resize(image, (RESIZED_IMG, RESIZED_IMG))
            image = torch.FloatTensor(np.transpose(image / 255., (2, 0, 1)))
            images.append(image)
        return images

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
        # 在1-len(images)范围选一张图片
        frame_idx = random.randint(1, len(video_data['image_files']) - 2)
        target_image, previous_image = self.load_frame_with_previous(video_name, frame_idx)
        # 创建mask
        mask = torch.zeros((target_image.shape[1], target_image.shape[2]))
        mask[:target_image.shape[1] // 2, :] = 1
        # 创建遮罩后的图像
        masked_image = target_image * mask
        # 获取对应音频即window中的音频
        audio_feature = self.load_audio_feature_with_window(video_name, frame_idx)
        #print(type(target_image))
        #print(type(previous_image))
        #print(type(masked_image))
        #print(type(audio_feature))
        return self.transform(target_image), self.transform(previous_image), self.transform(masked_image), audio_feature


if __name__ == "__main__":
    # get_face_mask(cv2.imread("./results/tjl/full_images/00000000.png"))
    val_data = MuseTalkDataset()
    dataloader = DataLoader(val_data, batch_size=1)
    print(len(dataloader))
    for i in dataloader:
        print(i)
        #ti, pi, mi, af = i
        #print(ti.shape)
        #print(mi.shape)
        #print(af.shape)
        #break
