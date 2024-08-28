from typing import Union, Iterable

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim, HOP_LENGTH
from whisper.model import Conv1d, ResidualAttentionBlock, LayerNorm, sinusoids


class AudioEncoder(nn.Module):
    def __init__(
            self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, include_embeddings: bool = True):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        if include_embeddings:
            embeddings = [x.cpu().detach().numpy()]
            for block in self.blocks:
                x = block(x)
                embeddings.append(x.cpu().detach().numpy())
            x = self.ln_post(x)
            return x, torch.Tensor(np.stack(embeddings, axis=1))
        else:
            for block in self.blocks:
                x = block(x)
            x = self.ln_post(x)
        return x


class AudioFeatureExtractor:

    def __init__(self, model_path, device, dtype):
        self.device = device
        self.dtype = dtype
        # 加载whisper的audio encoder
        state_dict = torch.load(model_path)
        dims = state_dict['dims']
        state_dict = state_dict['model_state_dict']
        encoder_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('encoder'):
                encoder_state_dict[key.replace('encoder.', '')] = state_dict[key]
        del state_dict
        self.encoder = AudioEncoder(
            n_mels=dims['n_mels'],
            n_ctx=dims['n_audio_ctx'],
            n_state=dims['n_audio_state'],
            n_head=dims['n_audio_head'],
            n_layer=dims['n_audio_layer']
        )
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder = self.encoder.to(device)

    @torch.no_grad()
    def extract_features(
            self,
            audio: Union[str, np.ndarray, torch.Tensor],
            audio_window=2
    ):
        mel = log_mel_spectrogram(audio)
        # 计算当sample_rate为16000时，对应的25fps的视频的总帧数
        frame_count = mel.shape[1] // 4
        features = []
        for start_idx in range(0, mel.shape[-1], N_FRAMES):
            mel_chunk = mel[:, start_idx: start_idx + 3000]
            segment = pad_or_trim(mel_chunk, N_FRAMES).to(self.device, dtype=self.dtype)
            single = segment.ndim == 2
            if single:
                segment = segment.unsqueeze(0)
            # embeddings的形状为n*5*1500*384,(n batch_size, 5 layers, 1500~30second, 384 embedding dim)
            # 单帧图像对应的音频特征为1 * 5 * 2 * 384
            _, embeddings = self.encoder(segment)
            features.append(embeddings.permute(0, 2, 1, 3))
        features = torch.cat(features, dim=1)

        hidden_dim = ((audio_window * 2) + 1) * 2 * 5
        audio_frame_features = torch.zeros((
            frame_count,
            hidden_dim,
            384
        ))
        for audio_idx in range(frame_count):
            start_idx = max(0, (audio_idx - audio_window) * 2)
            end_idx = min(frame_count * 2, (audio_idx + audio_window + 1) * 2)
            audio_frame_feature = features[0, start_idx:end_idx, :, :].reshape(1, -1, 384)

            # 对开始帧和结束帧进行填充
            if audio_frame_feature.shape[1] != hidden_dim:
                padding_feature = torch.zeros((1, hidden_dim - audio_frame_feature.shape[1], 384))
                if start_idx == 0:
                    audio_frame_feature = torch.cat([padding_feature, audio_frame_feature], dim=1)
                if end_idx == frame_count * 2:
                    audio_frame_feature = torch.cat([audio_frame_feature, padding_feature], dim=1)
            audio_frame_features[audio_idx] = audio_frame_feature
        return audio_frame_features


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    afe = AudioFeatureExtractor('../../models/whisper/tiny.pt', device=device, dtype=torch.float32)
    fs = afe.extract_features('../../data/audio/zack.wav', audio_window=0)
    print(fs.shape)
    # audio = whisper.audio.load_audio('../../data/audio/222.wav')
    # print(audio.shape)
    # chunks = afe.extract_frames(audio)
    # for chunk in chunks:
    #     print(chunk.shape)
