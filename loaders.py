import glob
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} speed: {(end - start) * 1000:.2f} ms")
        return result

    return inner


@timeit
def read_images_pil(filepath):
    print("reading images using PIL")
    arrays = []
    for file in tqdm(glob.glob(filepath + "/*")):
        img = Image.open(file)
        arrays.append(np.array(img))
    return arrays


@timeit
def read_images_cv(filepath):
    print("reading images using OpenCV")
    frames = []
    for img_path in tqdm(glob.glob(filepath + "/*")):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


if __name__ == '__main__':
    filedir = r"D:\Workplace\Applications\MuseTalkSimplify\results\avatars\avator_1\full_imgs"
    arrays = read_images_pil(filedir)
    del arrays
    frames = read_images_cv(filedir)
    del frames
