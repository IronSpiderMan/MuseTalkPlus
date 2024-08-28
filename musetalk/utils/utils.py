import os
import tempfile
import subprocess

import cv2
import torch
import edge_tts
import pandas as pd

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

from musetalk.models.vae import VAE
from musetalk.models.unet import UNet, PositionalEncoding
from musetalk.audio.feature_extractor import AudioFrameExtractor
from musetalk.audio.audio_feature_extractor import AudioFeatureExtractor
from common.setting import VAE_PATH, UNET_CONFIG_PATH, UNET_MODEL_PATH, WHISPER_PATH, WHISPER_FT_PATH

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


def load_all_model(afe: str):
    if afe.lower() == "musetalk":
        afe = AudioFeatureExtractor()
    elif afe.lower() == "custom":
        afe = AudioFrameExtractor(WHISPER_PATH)
    else:
        afe = AudioFrameExtractor(WHISPER_PATH)
    vae = VAE(model_path=VAE_PATH)
    unet = UNet(
        unet_config=UNET_CONFIG_PATH,
        model_path=UNET_MODEL_PATH,
        use_float16=True
    )
    pe = PositionalEncoding(d_model=384)
    # return audio_feature_extractor, vae, unet, pe
    return afe, vae, unet, pe


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
        audio_window=5,
        delay_frame=0,
):
    whisper_batch, latent_batch = [], []
    for i in range(audio_window, whisper_chunks.shape[0] - audio_window):
        idx = (i + delay_frame) % len(vae_encode_latents)
        latent = vae_encode_latents[idx]

        whisper_batch.append(whisper_chunks[i - audio_window: i + audio_window + 1, :, :].reshape(1, -1, 384))
        # print(whisper_batch[-1].shape)
        # whisper_batch.append(torch.cat([
        #     whisper_chunks[i - 1, :, :],
        #     whisper_chunks[i, :, :],
        #     whisper_chunks[i + 1, :, :],
        # ], dim=0)[None])
        # whisper_batch.append(whisper_chunks[i, i - 1:i + 2])
        latent_batch.append(latent)
        if len(latent_batch) >= batch_size:
            yield torch.cat(whisper_batch, dim=0), torch.cat(latent_batch, dim=0)
            whisper_batch, latent_batch = [], []

    if len(latent_batch) > 0:
        yield torch.cat(whisper_batch, dim=0), torch.cat(latent_batch, dim=0)

    # for i, w in enumerate(whisper_chunks):
    #     idx = (i + delay_frame) % len(vae_encode_latents)
    #     latent = vae_encode_latents[idx]
    #     whisper_batch.append(w)
    #     latent_batch.append(latent)
    #
    #     if len(latent_batch) >= batch_size:
    #         yield torch.from_numpy(np.stack(whisper_batch)), torch.cat(latent_batch, dim=0)
    #         whisper_batch, latent_batch = [], []
    #
    # # the last batch may smaller than batch size
    # if len(latent_batch) > 0:
    #     yield torch.from_numpy(np.stack(whisper_batch)), torch.cat(latent_batch, dim=0)


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
