import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class FaceParser:
    def __init__(self, model_path, device=torch.device('cuda')):
        self.device = device
        self.model_path = model_path
        self.image_processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)

    def __call__(self, image):
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # H x W
            mode='bilinear',
            align_corners=False
        )
        return upsampled_logits.argmax(dim=1)[0]


if __name__ == '__main__':
    fp = FaceParser('jonathandinu/face-parsing')
    out = fp(Image.open('1.jpg')).cpu().numpy()
    plt.imshow(out)
    plt.show()
