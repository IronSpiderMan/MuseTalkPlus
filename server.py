import os
import sys
import glob
import copy
import json
import queue
import pickle
import shutil
import asyncio

import cv2
import torch
import numpy as np
from tqdm import tqdm
from fastapi import FastAPI, WebSocket, BackgroundTasks
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

from musetalk.utils.utils import datagen, pronounce, video2images
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_images
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model

AVATAR_DIR = "./results/avatars"
loaded_avatar = None
audio_feature_extractor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许全部来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)


def make_multiple_dirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = os.path.join(AVATAR_DIR, avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_path, 'full_imgs')
        self.coords_path = os.path.join(self.avatar_path, 'coords.pkl')
        self.latents_out_path = os.path.join(self.avatar_path, 'latents.pt')
        self.video_out_path = os.path.join(self.avatar_path, 'vid_output')
        self.mask_out_path = os.path.join(self.avatar_path, 'mask')
        self.mask_coords_path = os.path.join(self.avatar_path, 'mask_coords.pkl')
        self.avatar_info_path = os.path.join(self.avatar_path, 'avator_info.json')
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.input_latent_list_cycle = None
        self.coord_list_cycle = None
        self.frame_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    make_multiple_dirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
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
                make_multiple_dirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
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
                    make_multiple_dirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
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
            video2images(self.video_path, self.full_imgs_path)
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

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
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def realtime_inference(
            self,
            audio_path,
            fps=25,
    ):
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        whisper_chunks = audio_feature_extractor.extract_and_chunk_feature(audio_path, fps)
        video_num = len(whisper_chunks)

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        for i, (whisper_batch, latent_batch) in enumerate(
                tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            whisper_batch = whisper_batch.to(
                device=unet.device,
                dtype=unet.model.dtype
            )
            whisper_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(
                latent_batch,
                0,
                encoder_hidden_states=whisper_batch
            ).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                current_idx = self.increase_idx()
                bbox = self.coord_list_cycle[current_idx]
                x1, y1, x2, y2 = bbox
                ori_frame = copy.deepcopy(self.frame_list_cycle[current_idx])
                # 必要时才Resize
                target_size = (x2 - x1, y2 - y1)
                if res_frame.shape[1] != target_size[0] or res_frame.shape[0] != target_size[1]:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                else:
                    res_frame = res_frame.astype(np.uint8)
                mask = self.mask_list_cycle[current_idx]
                mask_crop_box = self.mask_coords_list_cycle[current_idx]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                if i == 0:
                    inference_result_queue.put("start")
                inference_result_queue.put(combine_frame)
        inference_result_queue.put("end")

    def increase_idx(self):
        self.idx = (self.idx + 1) % len(self.frame_list_cycle)
        return self.idx

    async def next_frame(self, fps=25):
        inference = False
        while True:
            if not inference_result_queue.empty() or inference:
                flag = inference_result_queue.get()
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
            await asyncio.sleep(1 / fps)  # 控制发送图片的速度


avatar = Avatar(
    avatar_id="tjl",
    video_path="",
    bbox_shift=8,
    batch_size=4,
    preparation=False
)
inference_result_queue = queue.Queue()


async def get_compressed_image_data(image, max_width=450, max_height=450):
    img = Image.fromarray(image[:, :, ::-1])
    img.thumbnail((max_width, max_height))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)  # 调整质量以进一步压缩
    return buffer.getvalue()


@app.get("/talk")
async def talk(text: str, background_tasks: BackgroundTasks):
    filepath = await pronounce(text, "female", "US", stream=False)
    background_tasks.add_task(avatar.realtime_inference, filepath, 25)
    return {"data": text}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global avatar
    await websocket.accept()
    async for frame in avatar.next_frame(fps=25):
        image_data = await get_compressed_image_data(frame)
        await websocket.send_bytes(image_data)
