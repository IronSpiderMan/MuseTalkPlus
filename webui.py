import os

import torch
import gradio as gr

from musetalk.avatar import Avatar
from musetalk.utils.utils import load_all_model
from setting import AVATAR_DIR

whisper, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

avatar: Avatar


def inference(text):
    global avatar
    if not avatar:
        return None
    return avatar.inference(
        "",
        text,
    )


def load_avatar(avatar_id):
    global avatar
    if not avatar:
        avatar = Avatar(
            unet,
            vae,
            pe,
            whisper,
            avatar_id=avatar_id,
        )
    return "Avatar Loaded."


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                speech = gr.Textbox(label="请输入需要数字人说的内容")
                avatars = gr.Dropdown(os.listdir(AVATAR_DIR))
            output = gr.Textbox()
            load_btn = gr.Button('加载')
            load_generate = gr.Button('生成视频')
        with gr.Column():
            video = gr.Video(height=300, width=300)

    load_btn.click(load_avatar, inputs=[avatars], outputs=[output])
    load_generate.click(inference, inputs=[speech], outputs=[video])

if __name__ == '__main__':
    demo.launch()
