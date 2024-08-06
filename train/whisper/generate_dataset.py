"""
# 数据集目录结构
dataset_name
    - data
        - 000000.mp3
        - 000001.mp3
        ...
    - train.csv
    - test.csv
    - val.csv

# *.csv的结构
audio,text
path/to/audio_file.mp3,对应的文本内容
"""
import os
import re
import glob
import hashlib
import asyncio
import argparse
from pathlib import Path
from typing import List

import edge_tts
import aiofiles
import pandas as pd
from tqdm import tqdm

pattern = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def split_sentence(text, min_length=5):
    if not text.strip():
        return []
    slist = []
    for sentence in pattern.split(text):
        if pattern.match(sentence) and slist:
            slist[-1] += sentence
        elif sentence:
            if len(sentence) >= min_length:
                slist.append(sentence)
    return slist


async def tts(text, voice, output_dir):
    """
    Pass a text and voice to generate an audio file.
    The filename will be generated using the text and voice, so when the text and voice are the same,
    the generated audio file will also be the same.

    Parameters
    ----------
    text: The speech content
    voice: Voice's shortname, you can find it in voices.csv or edge_tts.list_voices()
    output_dir: The dir where to save the audio.

    Returns
    -------
    The audio file path

    """
    communicate = edge_tts.Communicate(text, voice)
    filename = f"{hashlib.sha256((text + voice).encode()).hexdigest()}.mp3"
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        return output_path
    await communicate.save(output_path)
    return output_path


async def generate_from_txt(txt_path, voices: List[str], output_dir="", min_length=5):
    """
    This function reads a txt file, splits the content into sentences, and then generates audio files for each sentence.

    Parameters
    ----------
    txt_path: Txt file path
    voices: List of voice names
    output_dir
    min_length: Minimum length of sentence

    Returns
    -------
    A generator that yields audio files for each sentence and sentence itself.

    """
    async with aiofiles.open(txt_path, 'r', encoding='utf-8') as fp:
        async for line in fp:
            sentences = split_sentence(str(line), min_length)
            for sentence in sentences:
                # 生成音频
                for voice in voices:
                    filepath = await tts(sentence, voice, output_dir)
                    if filepath:
                        yield filepath, sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        default="files/小王子.txt"
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='edgetts_zh',
    )
    return parser.parse_args()


async def main():
    # languages = ['zh-CN', 'en-US', 'en-GB']
    languages = ['zh-CN']
    args = parse_args()
    current_dir = Path(__file__).resolve()
    basedir = current_dir.parent.parent.parent
    audio_dir = basedir / 'datasets' / args.dataset_name / 'audio'
    os.makedirs(audio_dir, exist_ok=True)
    filepath = args.filepath
    voices = pd.read_csv(current_dir / 'voices.csv')
    selected_voices = voices[voices['Locale'].isin(languages)]['ShortName'].values
    files = []
    if not os.path.exists(filepath):
        print("This is a wrong path!")
        return
    elif os.path.isfile(filepath):
        files = [filepath]
    elif os.path.isdir(filepath):
        files = list(glob.glob(os.path.join(filepath, '*.txt')))
    datas = []
    for file in files:
        print("Now processing", file)
        progress_bar = tqdm()
        async for audio_file, sentence in generate_from_txt(file, selected_voices, output_dir=audio_dir):
            datas.append({
                "audio": audio_file,
                "text": sentence
            })
            progress_bar.update(1)
        progress_bar.close()
    pd.DataFrame(datas).drop_duplicates().to_csv(os.path.join(args.dataset_name, 'dataset.csv'), index=False)


if __name__ == '__main__':
    asyncio.run(main())
