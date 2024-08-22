from typing import Union, Iterable

import torch
import numpy as np
from tqdm import tqdm
from torch import nn, Tensor
import torch.nn.functional as F
from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES
from whisper.model import Conv1d, ResidualAttentionBlock, LayerNorm, sinusoids

from common.utils import timeit


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
    def transcribe(
            self,
            audio: Union[str, np.ndarray, torch.Tensor],
            verbose: bool
    ):
        mel = log_mel_spectrogram(audio)

        all_segments = []

        def add_segment(
                *, start: float, end: float, encoder_embeddings
        ):

            all_segments.append(
                {
                    "start": start,
                    "end": end,
                    "encoder_embeddings": encoder_embeddings,
                }
            )

        # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
        num_frames = mel.shape[-1]
        seek = 0
        sample_skip = 3000  #
        with tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
            while seek < num_frames:
                # seek是开始的帧数
                end_seek = min(seek + sample_skip, num_frames)
                segment = pad_or_trim(mel[:, seek:seek + sample_skip], N_FRAMES).to(self.device, dtype=self.dtype)

                single = segment.ndim == 2
                if single:
                    segment = segment.unsqueeze(0)
                if self.dtype == torch.float16:
                    segment = segment.half()
                _, embeddings = self.encoder(segment, include_embeddings=True)
                add_segment(
                    start=seek,
                    end=end_seek,
                    encoder_embeddings=embeddings,
                )
                seek += sample_skip

        return dict(segments=all_segments)

    def extract_feature(self, audio: Union[str, np.ndarray, torch.Tensor], verbose: bool = True):
        result = self.transcribe(audio, verbose)
        embed_list = []
        for emb in result['segments']:
            torch.permute(emb['encoder_embeddings'], (0, 2, 1, 3))
            encoder_embeddings = torch.permute(emb['encoder_embeddings'], (0, 2, 1, 3))[0]
            start_idx, end_idx = emb['start'], emb['end']
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        return torch.cat(embed_list, dim=0)

    def chunk_feature(self, feature, return_tensor, fps, feature_length=(2, 2)):
        whisper_chunks = []
        whisper_idx_multiplier = 50. / fps
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")
        while True:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature, selected_idx = self.get_sliced_feature(
                feature, i,
                feature_length,
                fps
            )
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx > len(feature):
                break
        if return_tensor:
            whisper_chunks = np.array(whisper_chunks)
            return torch.Tensor(whisper_chunks)
        return whisper_chunks

    @staticmethod
    def get_sliced_feature(feature, vid_idx, audio_feat_length=(2, 2), fps=35):
        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2
        idxes = np.clip(np.arange(left_idx, right_idx), 0, len(feature) - 1)
        selected_feature = feature[idxes].reshape(-1, 384)
        return selected_feature, idxes

    @timeit
    def extract_frames(self, audio_path, return_tensor=True):
        return self.chunk_feature(self.extract_feature(audio_path), return_tensor, 25)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    afe = AudioFeatureExtractor('../../models/whisper/tiny.pt', device=device, dtype=torch.float32)
    chunks = afe.extract_frames('../../data/audio/111.mp3')
    for chunk in chunks:
        print(chunk.shape)
