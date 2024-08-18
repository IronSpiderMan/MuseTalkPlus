import cv2
import torch
import numpy as np
from torchvision.transforms import transforms


class ImageProcessor:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    def __call__(self, image, half_mask=False) -> torch.Tensor:
        if isinstance(image, str):
            image = cv2.imread(image)
        if half_mask:
            image[image.shape[0] // 2:, :, :] = 0
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = self.transform(image)
        return image

    def de_process(self, image: torch.Tensor) -> np.ndarray:
        image = image * self.std + self.mean
        image = image * 255.0
        return image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
