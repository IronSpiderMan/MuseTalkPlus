import sys
import shutil
from typing import Any
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

sys.path.append('.')
from common.setting import settings
from common.utils import video2images, read_images
from musetalk_plus.utils import datagen, images2video
from musetalk_plus.models import MuseTalkModel
from musetalk_plus.processors import ImageProcessor
from musetalk_plus.faces.face_analysis import FaceAnalyst
from musetalk_plus.audio.feature_extractor import AudioFrameExtractor
from musetalk_plus.audio.audio_feature_extract import AudioFeatureExtractor


@torch.no_grad()
class Avatar:
    def __init__(
            self, avatar_id: str, video_path: str, bbox_shift_size: int = 5, device: Any = 'cuda',
            dtype=torch.float16, fixed_face=True, reliable_feature=True
    ):
        """
        avatar_id: avatar的唯一标识
        video_path: 视频路径
        """
        self.idx = 0
        self.avatar_id = avatar_id
        self.video_path = Path(video_path)
        self.bbox_shift_size = bbox_shift_size
        self.device = device
        self.dtype = dtype
        self.fixed_face = fixed_face
        self.vae = AutoencoderKL.from_pretrained(
            settings.models.vae_path, use_safetensors=False
        ).to(device, dtype=dtype)
        if reliable_feature:
            self.afe = AudioFeatureExtractor(settings.models.whisper_path, device, dtype)
        else:
            self.afe = AudioFrameExtractor(settings.models.whisper_fine_tuning_path, device=device, dtype=dtype)
        self.image_processor = ImageProcessor()
        self.unet = MuseTalkModel(settings.models.unet_path).to(device, dtype=dtype)
        self.face_analyst = None

        # 保存avatar相关文件的目录
        self.avatar_path = Path(settings.avatar.avatar_dir) / avatar_id
        self.full_images_path = self.avatar_path / 'full_images'
        self.full_masks_path = self.avatar_path / 'full_masks'
        self.latents_path = self.avatar_path / 'latents.npy'
        self.coords_path = self.avatar_path / 'coords.npy'
        self.mask_coords_path = self.avatar_path / 'mask_coords.npy'
        self.vid_output_path = self.avatar_path / 'vid_output'
        self.tmp_path = self.avatar_path / 'tmp'

        # 保存avatar相关数据
        self.frame_cycle = []
        self.input_latent_cycle = []
        self.coord_cycle = []
        self.mask_cycle = []

        # 其它属性
        self.audio_window = 5
        self.face_location = None
        self.default_location = [0, 0, 0, 0]

        # 初始化数字人需要的相关信息
        self.init_avatar()

    def init_avatar(self):
        if self.avatar_path.exists():
            # 加载frames、coord_cycle、input_latent_cycle
            frame_list = sorted(
                list(self.full_images_path.glob('*.[jpJP][pnPN]*[gG]'))
            )
            mask_list = sorted(
                list(self.full_masks_path.glob('*.[jpJP][pnPN]*[gG]'))
            )
            frame_list = read_images([str(file) for file in frame_list])
            mask_list = read_images([str(file) for file in mask_list], grayscale=True)

            self.frame_cycle = frame_list + frame_list[::-1]
            self.mask_cycle = mask_list + mask_list[::-1]
            self.coord_cycle = np.load(self.coords_path)
            self.input_latent_cycle = torch.tensor(np.load(self.latents_path))
        else:
            self.prepare_avatar()

    def shift_bbox(self, xyxy):
        x1, y1, x2, y2 = xyxy
        x1 -= self.bbox_shift_size
        x2 += self.bbox_shift_size
        y1 -= self.bbox_shift_size
        y2 += self.bbox_shift_size
        return np.array([x1, y1, x2, y2])

    def init_directories(self):
        self.avatar_path.mkdir()
        self.full_images_path.mkdir()
        self.full_masks_path.mkdir()
        self.vid_output_path.mkdir()
        self.tmp_path.mkdir()

    def validate_avatar(self):
        """
        validate if this avatar is valid
        a valid avatar should have directories named full_images, full_masks, vid_output
        and files named coord.npy, latents.npy
        """
        if not self.full_images_path.exists():
            pass

    def prepare_avatar(self):
        print("preparing avatar ...")
        self.face_analyst = FaceAnalyst(
            settings.models.dwpose_config_path, settings.models.dwpose_model_path
        )
        self.init_directories()
        video2images(self.video_path, self.full_images_path)
        input_image_list = sorted(
            list(self.full_images_path.glob('*.[jpJP][pnPN]*[gG]'))
        )
        frame_list = read_images([str(file) for file in input_image_list])
        mask_list = []
        coord_list = []
        face_latent_list = []
        # 检测人脸
        for idx, frame in tqdm(enumerate(frame_list), desc="Detecting faces", total=len(frame_list)):
            pts = self.face_analyst.analysis(frame)
            h, w = frame.shape[:2]
            landmark_mask = self.face_analyst.face_landmark_mask((w, h), pts)
            bbox = self.face_analyst.face_location(pts)
            mask_list.append(landmark_mask)
            coord_list.append(bbox)
            cv2.imwrite(str(self.full_masks_path / f'{idx:08d}.jpg'), landmark_mask)

        for idx, (frame, coord) in tqdm(
                enumerate(zip(frame_list, coord_list)),
                desc='Encode face image',
                total=len(frame_list)
        ):
            x1, y1, x2, y2 = coord
            face = frame[y1:y2, x1:x2, :]
            # 编码人物形象图片
            avatar_face = self.image_processor(face)[None].to(self.device, dtype=self.dtype)
            avatar_face_latent = self.vae.encode(avatar_face).latent_dist.sample()
            avatar_face_latent = avatar_face_latent * self.vae.config.scaling_factor
            # 编码当前帧masked图片
            masked_face = self.image_processor(face.copy(), half_mask=True)[None].to(self.device, dtype=self.dtype)
            masked_latents = self.vae.encode(masked_face).latent_dist.sample()
            masked_latents = masked_latents * self.vae.config.scaling_factor
            # unet模型输入形状为n*8*32*32,其中n*0:4*32*32为人物图像，n*4:8*32*32为当前帧的masked图像
            latents = torch.cat([masked_latents, avatar_face_latent], dim=1)
            face_latent_list.append(latents.cpu().numpy())
        self.frame_cycle = frame_list + frame_list[::-1]
        self.mask_cycle = mask_list + mask_list[::-1]
        self.coord_cycle = np.array(coord_list + coord_list[::-1])
        self.input_latent_cycle = torch.tensor(np.concatenate(face_latent_list + face_latent_list[::-1], axis=0))

        # 保存相关信息
        np.save(self.coords_path, self.coord_cycle)
        np.save(self.latents_path, self.input_latent_cycle.numpy())

        del self.face_analyst

    @torch.no_grad()
    def inference(self, audio_path, batch_size=4):
        self.vid_output_path.mkdir(exist_ok=True)
        self.tmp_path.mkdir(exist_ok=True)
        frame_idx = self.idx = 0
        whisper_chunks = self.afe.extract_frames(audio_path, return_tensor=True)
        # gen = datagen(whisper_chunks, self.input_latent_cycle, 4, delay_frames=self.idx, audio_window=self.audio_window)
        gen = datagen(
            whisper_chunks, self.input_latent_cycle, frame_feature=False,
            batch_size=batch_size, delay_frames=self.idx, audio_window=self.audio_window
        )
        for i, (whisper_batch, latent_batch) in enumerate(
                tqdm(gen, total=whisper_chunks.shape[0] // batch_size, desc='Inference...')
        ):
            whisper_batch = whisper_batch.to(self.device, dtype=self.dtype)
            latent_batch = latent_batch.to(self.device, dtype=self.dtype)
            pred_latents = self.unet((latent_batch, whisper_batch))

            # # 测试帧和latents对应情况
            # pred_latents = latent_batch[:, 4:, :, :]
            pred_latents = (1 / self.vae.config.scaling_factor) * pred_latents
            pred_images = self.vae.decode(pred_latents).sample
            for idx, pred_image in enumerate(pred_images.cpu()):
                x1, y1, x2, y2 = self.coord_cycle[frame_idx]
                frame = self.frame_cycle[frame_idx].copy()
                resized_image = cv2.resize(self.image_processor.de_process(pred_image), (x2 - x1, y2 - y1))
                # 融合预测图像与原图像
                pil_frame = Image.fromarray(frame[:, :, ::-1])
                pil_face = Image.fromarray(resized_image[:, :, ::-1])
                pil_mask = Image.fromarray(self.mask_cycle[frame_idx]).convert('L').crop((x1, y1, x2, y2))
                pil_frame.paste(pil_face, box=[x1, y1, x2, y2], mask=pil_mask)
                # frame[y1:y2, x1:x2, :] = resized_image
                pil_frame.save(str(self.tmp_path / f'{frame_idx:08d}.jpg'))
                # cv2.imwrite(str(self.tmp_path / f'{frame_idx:08d}.jpg'), frame)
                frame_idx += 1
                self.idx += 1
        images2video(self.tmp_path, self.vid_output_path / (Path(audio_path).stem + '.mp4'))
        shutil.rmtree(self.tmp_path)

    def increase_idx(self):
        self.idx = self.idx + 1 % len(self.frame_cycle)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avatar = Avatar('111', r'F:\Workplace\MuseTalkPlus\data\video\zack.mp4', device=device, reliable_feature=True)
    # avatar.inference('./data/audio/out.mp3')
    # avatar.inference(r'F:\Workplace\MuseTalkPlus\data\audio\zack.mp3')
    avatar.inference(r'F:\Workplace\MuseTalkPlus\data\audio\00000002.mp3')


if __name__ == '__main__':
    main()
