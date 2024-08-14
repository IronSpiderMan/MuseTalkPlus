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
UNET_PATH = BASE_DIR / "models/musetalk"
UNET_CONFIG_PATH = UNET_PATH / "musetalk.json"
UNET_MODEL_PATH = UNET_PATH / "pytorch_model.bin"
VAE_PATH = BASE_DIR / "models/sd-vae-ft-mse"
DWPOST_PATH = BASE_DIR / "models/dwpose/dw-ll_ucoco_384.pth"

# 训练相关
TRAIN_OUTPUT_DIR = BASE_DIR / "outputs/ckpts"
TRAIN_OUTPUT_LOGS_DIR = BASE_DIR / "outputs/logs"
