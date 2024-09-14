import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import torch
import librosa
import soundfile

sys.path.append('svc')

import gradio as gr

from common.utils import tts
from musetalk.avatar import Avatar
from common.setting import settings
from svc.inference.infer_tool import Svc

avatar: Optional[Avatar] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
svc: Optional[Svc] = None

speaker_mapping = {
    "sun": "zh-CN-XiaoxiaoNeural",
}


def svc_tts(text: str, speaker: str):
    global svc
    print(speaker)
    filename = asyncio.run(tts(text, speaker_mapping[speaker]))
    target_sr = 44100
    y, sr = librosa.load(filename)
    resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    tmp_path = f'tmp/{uuid.uuid4()}.wav'
    soundfile.write(tmp_path, resampled_y, target_sr, subtype="PCM_16")
    # 转换音色
    _audio = svc.slice_inference(
        tmp_path,
        0,
        0,
        -40,
        0,
        True,
        0.4,
        f0_predictor="harvest"
    )
    svc.clear_empty()
    dst_path = f"tmp/{uuid.uuid4()}.wav"
    soundfile.write(dst_path, _audio, svc.target_sample, format='wav')
    os.remove(tmp_path)
    return dst_path.replace('"', "")


def inference(text):
    if not avatar:
        return None
    fpath = svc_tts(text, avatar.avatar_id)
    return avatar.inference(
        fpath,
    )


def load_avatar(avatar_id):
    global avatar, svc
    if avatar is None:
        avatar = Avatar(str(avatar_id), 'video_path', 5, device)
    if svc is None:
        speaker_path = Path('speakers') / avatar_id
        config_path = str(list(speaker_path.glob('*.json'))[0])
        model_path = str(list(speaker_path.glob("*.pth"))[0])
        svc = Svc(
            model_path,
            config_path,
            device='cuda',
            nsf_hifigan_enhance=False,
            shallow_diffusion=False,
            only_diffusion=False,
            spk_mix_enable=False,
            feature_retrieval=None
        )
    return "Avatar Loaded."


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                speech = gr.Textbox(label="请输入需要数字人说的内容")
                avatars = gr.Dropdown([
                    str(file.name) for file in Path(settings.avatar.avatar_dir).iterdir()
                ])
            output = gr.Textbox()
            load_btn = gr.Button('加载')
            load_generate = gr.Button('生成视频')
        with gr.Column():
            video = gr.Video(height=300, width=300)

    load_btn.click(load_avatar, inputs=[avatars], outputs=[output])
    load_generate.click(inference, inputs=[speech], outputs=[video])

if __name__ == '__main__':
    demo.launch()
