import os
import sys
import glob
import json
import time
import copy
import queue
import pickle
import shutil
import asyncio
import threading
from typing import Literal

import cv2
import torch
import numpy as np
from tqdm import tqdm

from common.setting import AVATAR_DIR
from musetalk.utils.utils import pronounce, datagen
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from musetalk.whisper.audio_feature_extractor import AudioFeatureExtractor
from common.utils import make_multiple_dirs, read_images, video2images


@torch.no_grad()
class Avatar:
    def __init__(
            self,
            unet: UNet,
            vae: VAE,
            pe: PositionalEncoding,
            whisper: AudioFeatureExtractor,
            avatar_id: str,
            video_path: str = '',
            fps: int = 25,
            bbox_shift: int = 8,
            batch_size: int = 4,
            preparation: bool = False,
    ):
        self.unet = unet
        self.vae = vae
        self.pe = pe
        self.whisper = whisper
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.fps = fps
        self.avatar_path = AVATAR_DIR / avatar_id
        self.full_images_path = self.avatar_path / 'full_images'
        self.coords_path = self.avatar_path / 'coords.pkl'
        self.latents_out_path = self.avatar_path / 'latents.pt'
        self.video_out_path = self.avatar_path / 'vid_output'
        self.mask_out_path = self.avatar_path / 'mask'
        self.mask_coords_path = self.avatar_path / 'mask_coords.pkl'
        self.avatar_info_path = self.avatar_path / 'avatar_info.json'
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.idx = 0
        self.preparation = preparation
        self.batch_size = batch_size
        self.input_latent_list_cycle = None
        self.coord_list_cycle = None
        self.frame_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None
        self.inference_result_queue = queue.Queue()
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    make_multiple_dirs(
                        [self.avatar_path, self.full_images_path, self.video_out_path, self.mask_out_path]
                    )
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_images_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_images(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list,
                                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_images(input_mask_list, grayscale=True)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                make_multiple_dirs([self.avatar_path, self.full_images_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    make_multiple_dirs(
                        [self.avatar_path, self.full_images_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_images_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_images(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_images(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2images(self.video_path, self.full_images_path)
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_images_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_images_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_images_path}/{str(i).zfill(8)}.png", frame)

            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, self.latents_out_path)

    def inference(
            self,
            audio_path: str,
            text: str,
            out_vid_name: str = None,
            fps: int = 25,
            realtime=False
    ):
        os.makedirs(self.avatar_path / 'tmp', exist_ok=True)
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()

        if not audio_path:
            audio_path = asyncio.run(pronounce(text, "female", "US", stream=False))
        if not fps:
            fps = self.fps
        whisper_chunks = self.whisper.extract_and_chunk_feature(audio_path, fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        self.idx = 0
        gen = datagen(
            whisper_chunks,
            self.input_latent_list_cycle,
            self.batch_size
        )
        frame_idx = 0
        for i, (whisper_batch, latent_batch) in enumerate(
                tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            whisper_batch = whisper_batch.to(
                device=self.unet.device,
                dtype=self.unet.model.dtype
            )
            whisper_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(
                latent_batch,
                0,
                encoder_hidden_states=whisper_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                combine_frame = self.process_frame(
                    res_frame,
                    "start" if frame_idx == 0 else None if frame_idx == len(
                        recon) - 1 else "end"
                )
                if not realtime:
                    cv2.imwrite(str(self.avatar_path / "tmp" / f"{str(frame_idx).zfill(8)}.png"), combine_frame)
                frame_idx += 1

        if out_vid_name and not realtime:
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = self.video_out_path / f"{out_vid_name}.mp4"
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(self.avatar_path / "temp.mp4")
            # shutil.rmtree(self.avatar_path / "tmp")
            print(f"result is save to {output_vid}")
        print("\n")

    def process_frame(self, frame, flag: Literal['start', 'end'] = None):
        bbox = self.coord_list_cycle[self.idx]
        x1, y1, x2, y2 = bbox
        ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx])
        # 必要时才Resize
        target_size = (x2 - x1, y2 - y1)
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            res_frame = cv2.resize(frame.astype(np.uint8), (x2 - x1, y2 - y1))
        else:
            res_frame = frame.astype(np.uint8)
        mask = self.mask_list_cycle[self.idx]
        mask_crop_box = self.mask_coords_list_cycle[self.idx]
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        if flag == 'start':
            self.inference_result_queue.put(flag)
        self.inference_result_queue.put(combine_frame)
        if flag == 'end':
            self.inference_result_queue.put(flag)
        self.increase_idx()
        return combine_frame

    def increase_idx(self):
        self.idx = (self.idx + 1) % len(self.frame_list_cycle)
        return self.idx

    async def next_frame(self):
        inference = False
        while True:
            if not self.inference_result_queue.empty() or inference:
                flag = self.inference_result_queue.get()
                if isinstance(flag, str):
                    if flag == "start":
                        inference = True
                    else:
                        inference = False
                    continue
                yield flag
            else:
                self.increase_idx()
                yield self.frame_list_cycle[self.idx]
            await asyncio.sleep(1 / self.fps)  # 控制发送图片的速度
