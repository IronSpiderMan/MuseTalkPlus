from torch import nn
from diffusers import UNet2DConditionModel


class MuseTalkModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_path, use_safetensors=False)

    def forward(self, inputs):
        image_feature, audio_feature = inputs
        return self.unet(image_feature, 0, encoder_hidden_states=audio_feature).sample
