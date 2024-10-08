import os
import time
import shutil
import subprocess
from uuid import uuid4
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import cv2
import edge_tts
from tqdm import tqdm


def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} speed: {(end - start) * 1000:.2f} ms")
        return result

    return inner


def video2images(vid_path, save_path):
    output_pattern = os.path.join(save_path, "%08d.png")
    ffmpeg_command = [
        "ffmpeg",
        "-i", vid_path,
        "-vf", "fps=25",
        output_pattern
    ]
    try:
        subprocess.run(ffmpeg_command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
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
        subprocess.run(ffmpeg_command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print(f"Frames extracted to {save_path}")
        return save_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def make_multiple_dirs(path_list, exist_ok=True):
    for path in path_list:
        os.makedirs(path, exist_ok=exist_ok) if not os.path.exists(path) else None


def remove_multiple_dirs(path_list):
    for path in path_list:
        # os.removedirs(path) if os.path.exists(path) else None
        shutil.rmtree(path) if os.path.exists(path) else None


def recreate_multiple_dirs(path_list):
    remove_multiple_dirs(path_list)
    make_multiple_dirs(path_list)


def read_images(img_list, grayscale=False, to_rgb=True):
    """
    根据图像的文件列表，使用多线程读取图像，返回图像RGB模式的ndarray列表
    """
    frames = []
    with ThreadPoolExecutor() as executor:
        # Use partial to fix the flags parameter for cv2.imread
        if grayscale:
            imread = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        else:
            imread = cv2.imread
        for frame in tqdm(executor.map(imread, img_list), total=len(img_list), desc='Reading images'):
            if not grayscale and to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


async def tts(message, voice="zh-CN-XiaoxiaoNeural"):
    voice = edge_tts.Communicate(text=message, voice=voice, rate='-4%', volume='+0%')
    dst_file = f"tmp/{uuid4().hex}.wav"
    await voice.save(dst_file)
    return dst_file
