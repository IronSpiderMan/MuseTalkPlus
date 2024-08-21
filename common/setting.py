from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset相关目录
DATASET_DIR = BASE_DIR / 'datasets'
VIDEO_DIR = DATASET_DIR / "video"  # 存放数据集原始视频
TMP_DATASET_DIR = DATASET_DIR / "tmp"
AUDIO_FEATURE_DIR = DATASET_DIR / "audios"
VIDEO_FRAME_DIR = DATASET_DIR / "images"
VIDEO_LATENT_DIR = DATASET_DIR / "latents"

TMP_AUDIO_DIR = TMP_DATASET_DIR / "audios"
TMP_FRAME_DIR = TMP_DATASET_DIR / "images"

# avatar相关目录
AVATAR_DIR = BASE_DIR / "results"

# 模型相关
WHISPER_PATH = BASE_DIR / "models/whisper/tiny.pt"
WHISPER_FT_PATH = BASE_DIR / "models/whisper-tiny-zh"
UNET_PATH = BASE_DIR / "models/musetalk"
UNET_CONFIG_PATH = UNET_PATH / "musetalk.json"
UNET_MODEL_PATH = UNET_PATH / "diffusion_pytorch_model.bin"
VAE_PATH = BASE_DIR / "models/sd-vae-ft-mse"
DWPOST_PATH = BASE_DIR / "models/dwpose/dw-ll_ucoco_384.pth"
DWPOSE_CONFIG_PATH = BASE_DIR / "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"

# 训练相关
TRAIN_OUTPUT_DIR = BASE_DIR / "outputs/ckpts"
TRAIN_OUTPUT_LOGS_DIR = BASE_DIR / "outputs/logs"

from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    videos_dir: str
    audios_dir: str
    images_dir: str
    latents_dir: str
    audio_window: int


@dataclass
class TrainConfig:
    output: str


@dataclass
class AvatarConfig:
    avatar_dir: str


@dataclass
class ModelsConfig:
    whisper_path: str
    whisper_fine_tuning_path: str
    unet_path: str
    vae_path: str
    dwpose_config_path: str
    dwpose_model_path: str


@dataclass
class Settings:
    dataset: DatasetConfig
    train: TrainConfig
    avatar: AvatarConfig
    models: ModelsConfig

    @classmethod
    def from_yaml(cls, file_path: str) -> "Settings":
        # 加载 YAML 文件并转换为字典
        config_dict = OmegaConf.load(file_path)
        # 将字典转换为 OmegaConf 对象的结构化形式
        config_struct = OmegaConf.structured(cls)
        # 合并加载的配置到结构化对象中
        config = OmegaConf.merge(config_struct, config_dict)
        # 转换为 Settings 实例
        return OmegaConf.to_object(config)

    def save_to_yaml(self, file_path: str):
        # 将当前配置保存为 YAML 文件
        config_dict = OmegaConf.structured(self)
        OmegaConf.save(config_dict, file_path)


settings = Settings.from_yaml(str(Path(__file__).parent / "settings.yaml"))
