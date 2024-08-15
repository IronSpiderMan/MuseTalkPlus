import math

import numpy as np
import torch
import librosa
from transformers import WhisperProcessor, WhisperModel


class AudioFrameExtractor:
    def __init__(self, model_name_or_path):
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.model = WhisperModel.from_pretrained(model_name_or_path)
        self.sample_rate = 16000
        self.video_fps = 25
        self.audio_fps = self.sample_rate // self.video_fps

    def extract_frames(self, audio_path, return_tensor=False):
        audio, sr = librosa.load(audio_path, sr=16000)
        # 计算视频总帧数
        frames = math.ceil(audio.shape[-1] / self.audio_fps)
        input_features = self.processor(audio, sampling_rate=sr, return_tensors='pt').input_features
        if return_tensor:
            segments = torch.zeros((frames, 2, 384))
        else:
            segments = np.zeros((frames, 2, 384))
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            # audio_features形状为n × 1500 × 384
            audio_features = encoder_outputs.last_hidden_state
            for i in range(frames):
                start = i * 2
                end = start + 2
                segments[i, :, :] = audio_features[0, start:end, :]
        return segments


if __name__ == '__main__':
    # 加载音频文件
    audio_file = r"F:\Workplace\MuseTalkPlus\data\audio\out_.mp3"
    afe = AudioFrameExtractor(model_name_or_path=r"F:\models\whisper-tiny-zh")
    print(afe.extract_frames(audio_file).shape)
