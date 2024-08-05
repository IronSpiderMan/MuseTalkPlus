# import torch
# from musetalk.models.vae import VAE
#
# vae = VAE()
# img = torch.rand(4, 3, 256, 256).to("cuda")
# latents = vae.encode_latents(img)
# print(latents.shape)
import torch
from musetalk.models.unet import UNet

unet = UNet(
    unet_config="./models/musetalk/musetalk.json",
    model_path="./models/musetalk/pytorch_model.bin",
)
i = torch.rand(1, 8, 32, 32).to("cuda")
j = torch.rand(1, 50, 384).to("cuda")
latents = unet.model(i, 0, encoder_hidden_states=j)
print(latents.sample.shape)
