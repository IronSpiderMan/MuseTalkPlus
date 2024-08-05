import queue
from functools import partial
from io import BytesIO

import torch
from fastapi import FastAPI, WebSocket, BackgroundTasks
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

from musetalk.avatar import Avatar
from musetalk.utils.utils import load_all_model

AVATAR_DIR = "./results/avatars"
loaded_avatar = None
whisper, vae, unet, pe = load_all_model()
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

avatar = Avatar(
    unet,
    vae,
    pe,
    whisper,
    avatar_id="tjl",
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
    inference = partial(avatar.inference, {"audio_path": "", "text": text})
    background_tasks.add_task(inference)
    return {"data": text}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global avatar
    await websocket.accept()
    async for frame in avatar.next_frame():
        image_data = await get_compressed_image_data(frame)
        await websocket.send_bytes(image_data)
