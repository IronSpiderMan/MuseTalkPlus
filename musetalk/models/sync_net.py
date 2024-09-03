import torch
from torch import nn

sync_t = 5


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 输入图像序列, batch_size * 3 * 256 * 256
        self.face_encoder = nn.Sequential(
            Conv2d(3 * sync_t, 32, kernel_size=7, stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
        )
        # 输入音频序列, batch_size * 1 * 50 * 384
        self.audio_encoder = nn.Sequential(
            Conv2d(1 * sync_t, 32, kernel_size=7, stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
        )

    def forward(self, inputs):
        images, audios = inputs
        image_embeddings = self.face_encoder(images)
        audio_embeddings = self.audio_encoder(audios)
        return image_embeddings, audio_embeddings


if __name__ == '__main__':
    syncnet = SyncNet()
    i = (torch.rand(1, 3 * sync_t, 256, 256), torch.rand(1, 1 * sync_t, 50, 384))
    o = syncnet(i)
    print(o[0].shape)
    print(o[1].shape)
