from pathlib import Path

# dataset相关目录
DATASET_DIR = Path("./datasets")
VIDEO_DIR = DATASET_DIR / "video"  # 存放数据集原始视频
TMP_DATASET_DIR = DATASET_DIR / "tmp"
AUDIO_FEATURE_DIR = DATASET_DIR / "audios"
VIDEO_FRAME_DIR = DATASET_DIR / "images"

TMP_AUDIO_DIR = TMP_DATASET_DIR / "audios"
TMP_FRAME_DIR = TMP_DATASET_DIR / "images"

# avatar相关目录
AVATAR_DIR = Path("./results")

# 模型相关
WHISPER_PATH = Path("./models/whisper/tiny.pt")
UNET_PATH = Path("./models/musetalk")
UNET_CONFIG_PATH = UNET_PATH / "musetalk.json"
VAE_PATH = Path("./models/sd-vae-ft-mse")
DWPOST_PATH = Path("./models/dwpose/dw-ll_ucoco_384.pth")

# 训练相关
TRAIN_OUTPUT_DIR = Path("./outputs/ckpts")
