import os

import cv2
import numpy as np
import torch
import edge_tts
import pandas as pd
import tempfile
import subprocess


ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

from musetalk.whisper.audio_feature_extractor import AudioFeatureExtractor
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding

voices = pd.DataFrame([
    {
        "Gender": "female",
        "Style": "US",
        "Name": "en-US-AvaNeural"
    },
    {
        "Gender": "male",
        "Style": "US",
        "Name": "en-US-AndrewNeural"
    },
    {
        "Gender": "female",
        "Style": "GB",
        "Name": "en-GB-LibbyNeural"
    },
    {
        "Gender": "male",
        "Style": "GB",
        "Name": "en-GB-RyanNeural"
    }
])


def load_all_model():
    audio_feature_extractor = AudioFeatureExtractor(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path="./models/sd-vae-ft-mse/")
    unet = UNet(
        unet_config="./models/musetalk/musetalk.json",
        model_path="./models/musetalk/pytorch_model.bin",
        use_float16=True
    )
    pe = PositionalEncoding(d_model=384)
    return audio_feature_extractor, vae, unet, pe


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'


def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def datagen(
        whisper_chunks,
        vae_encode_latents,
        batch_size=8,
        delay_frame=0
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i + delay_frame) % len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            yield torch.from_numpy(np.stack(whisper_batch)), torch.cat(latent_batch, dim=0)
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        yield torch.from_numpy(np.stack(whisper_batch)), torch.cat(latent_batch, dim=0)


async def pronounce(text, gender, style, save_path="tmp", stream=True):
    voice = voices[(voices['Gender'] == gender) & (voices['Style'] == style)]['Name'].values[0]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=save_path) as fp:
        voice = edge_tts.Communicate(text=text, voice=voice, rate='-4%', volume='+0%')
        if stream:
            audio_stream = voice.stream()
            return audio_stream
        else:
            await voice.save(fp.name)
            return fp.name


def video2images(vid_path, save_path, fps=26):
    filename = os.path.basename(vid_path).split(".")[0]
    save_path = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True)
    output_pattern = os.path.join(str(save_path), "%08d.png")
    ffmpeg_command = [
        "ffmpeg",
        "-i", vid_path,
        "-vf", f"fps={fps}",
        output_pattern
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Frames extracted to {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def video2audio(vid_path, save_path):
    filename = os.path.basename(vid_path).split(".")[0] + ".mp3"
    save_path = os.path.join(save_path, filename)
    ffmpeg_command = [
        "ffmpeg",
        "-i", vid_path,
        "-q:a", "0",
        "-map", "a",
        save_path, "-y"
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Frames extracted to {save_path}")
        return save_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
