# import os
# DATASET_DIR = "./datasets"
# TMP_DATASET_DIR = os.path.join(DATASET_DIR, 'tmp')
# AUDIO_FEATURE_DIR = os.path.join(DATASET_DIR, "audios")
# VIDEO_FRAME_DIR = os.path.join(DATASET_DIR, "images")
#
# TMP_AUDIO_DIR = os.path.join(TMP_DATASET_DIR, "audios")
# TMP_FRAME_DIR = os.path.join(TMP_DATASET_DIR, "images")
#
# AVATAR_DIR = "./results"
from pathlib import Path

# dataset相关目录
DATASET_DIR = Path("./datasets")
TMP_DATASET_DIR = DATASET_DIR / "tmp"
AUDIO_FEATURE_DIR = DATASET_DIR / "audios"
VIDEO_FRAME_DIR = DATASET_DIR / "images"

TMP_AUDIO_DIR = TMP_DATASET_DIR / "audios"
TMP_FRAME_DIR = TMP_DATASET_DIR / "images"

# avatar相关目录
AVATAR_DIR = Path("./results")
