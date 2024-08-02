import json
import torch
import tqdm
from torch import optim
from diffusers import AutoencoderKL, UNet2DConditionModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys

from triton.language import dtype

sys.path.append('.')
from musetalk.models.unet import PositionalEncoding
from myloader import MuseTalkDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# 加载数据
train_dataset = MuseTalkDataset()
# train_loader = DataLoader(train_dataset, collate_fn=lambda x: x[0])
train_loader = DataLoader(train_dataset, batch_size=8)
# 加载模型
vae = AutoencoderKL.from_pretrained("./models/sd-vae-ft-mse", subfolder="vae").to(device)
with open("./models/musetalk/musetalk.json", "r") as f:
    unet_config = json.load(f)
unet = UNet2DConditionModel(**unet_config).to(device)
pe = PositionalEncoding(d_model=384).to(device)
# 创建优化器
optimizer = optim.AdamW(
    unet.parameters(),
    lr=5e-6,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08
)
for idx, (target_image, ref_image, masked_image, mask, audio_feature) in tqdm.tqdm(enumerate(train_loader)):
    # target_image = preprocess_img_tensor(target_image).to(device)
    # ref_image = preprocess_img_tensor(ref_image).to(device)
    # masked_image = preprocess_img_tensor(masked_image).to(device)
    target_image = transform(target_image).to(device, dtype=torch.float32)
    ref_image = transform(ref_image).to(device, dtype=torch.float32)
    masked_image = transform(masked_image).to(device, dtype=torch.float32)

    latents = vae.encode(target_image).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    masked_latents = vae.encode(masked_image).latent_dist.sample()
    masked_latents = masked_latents * vae.config.scaling_factor

    ref_latents = vae.encode(ref_image).latent_dist.sample()
    ref_latents = ref_latents * vae.config.scaling_factor

    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

    image_pred = unet(latent_model_input, 0, encoder_hidden_states=audio_feature.to(device)).sample

    loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    logs = {"loss": loss.detach().item()}
    # 保存模型参数
    print(logs)
    if (idx + 1) % 10 == 0:
        torch.save(unet.state_dict(), f'musetalk--{idx}.pth')
