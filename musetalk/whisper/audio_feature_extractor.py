import torch
import numpy as np

from common.utils import timeit
from common.setting import WHISPER_PATH
from musetalk.whisper.whisper import Whisper, ModelDimensions


class AudioFeatureExtractor:
    def __init__(
            self,
            model_path=WHISPER_PATH,
            device="auto"
    ):
        self.device = self.auto_device(device)
        self.model_path = model_path
        self.model = self.load_model()

    def auto_device(self, device):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cpu", "cuda", "mps"]:
            self.device = device
        else:
            self.device = "cpu"
        return self.device

    def load_model(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        dims = ModelDimensions(**ckpt["dims"])
        model = Whisper(dims)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(self.device)

    def extract_frames(self, audio_path, return_tensor=True):
        if return_tensor:
            return torch.tensor(self.extract_and_chunk_feature(audio_path, 25))
        else:
            return self.extract_and_chunk_feature(audio_path, 25)

    @timeit
    def extract_and_chunk_feature(self, audio_path, fps=26):
        return self.chunk_feature(self.extract_feature(audio_path), fps)

    def extract_feature(self, audio_path):
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings'].transpose(0, 2, 1, 3).squeeze(0)
            start_idx, end_idx = emb['start'], emb['end']
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        return np.concatenate(embed_list, axis=0)

    def chunk_feature(self, feature, fps, feature_length=(2, 2)):
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
        return whisper_chunks

    @staticmethod
    def get_sliced_feature(feature, vid_idx, audio_feat_length=(2, 2), fps=26):
        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2
        idxes = np.clip(np.arange(left_idx, right_idx), 0, len(feature) - 1)
        selected_feature = feature[idxes].reshape(-1, 384)
        return selected_feature, idxes


if __name__ == '__main__':
    fe = AudioFeatureExtractor()
    fe.extract_and_chunk_feature("./data/audio/00000002.mp3", 25)
