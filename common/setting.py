from pathlib import Path

from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class CommonConfig:
    fps: int
    image_size: int
    hidden_size: int
    embedding_dim: int


@dataclass
class DatasetConfig:
    base_dir: str
    videos_dir: str
    audios_dir: str
    images_dir: str
    latents_dir: str
    audio_window: int


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    audio_window: int
    related_window: int
    gamma: float
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
    common: CommonConfig
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
