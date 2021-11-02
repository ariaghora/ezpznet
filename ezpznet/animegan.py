import os
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ezpznet import EzPzNet, download_weight


URL_WEBTOON = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131925&authkey=AFr6EizN764A3u0"
URL_SHINKAI = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131929&authkey=ANQyvelJifvCiU8"
URL_PAPRIKA = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131930&authkey=ACbm3QKgoAA5w3k"
URL_HAYAO = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131928&authkey=ALgdQbAiArkG2Sg"


class ConvNormLReLU(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        pad_mode="reflect",
        groups=1,
        bias=False,
    ):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(
            ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True)
        )
        # pw
        layers.append(
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False)
        )
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64),
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128),
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(ConvNormLReLU(128, 128), ConvNormLReLU(128, 128))

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=False
            )
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(
                out, input.size()[-2:], mode="bilinear", align_corners=True
            )
        else:
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=False
            )
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


def load_image(image_path: str, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32:  # resize image to multiple of 32s

        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img / 127.5 - 1.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


class AnimeGAN(EzPzNet):
    def __init__(self, style: str = "webtoon"):
        super().__init__()
        self.style = style

        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        weights_name_webtoon = os.path.join(weights_dir, "face_paint_512_v2_0.pt")
        weights_name_shinkai = os.path.join(weights_dir, "shinkai.pt")
        weights_name_paprika = os.path.join(weights_dir, "paprika.pt")
        weights_name_hayao = os.path.join(weights_dir, "hayao.pt")

        if self.style == "webtoon":
            download_weight(URL_WEBTOON, weights_name_webtoon)
            weight_path = weights_name_webtoon
        elif self.style == "shinkai":
            download_weight(URL_SHINKAI, weights_name_webtoon)
            weight_path = weights_name_shinkai
        elif self.style == "paprika":
            download_weight(URL_PAPRIKA, weights_name_webtoon)
            weight_path = weights_name_paprika
        elif self.style == "hayao":
            download_weight(URL_HAYAO, weights_name_webtoon)
            weight_path = weights_name_hayao
        else:
            raise Exception("Style is not supported")

        self.net = Generator()
        self.net.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.net.to(self.device).eval()

    def predict(self, X):
        with torch.no_grad():
            X = X.to(self.device)
            out = (
                self.net(X, align_corners=False)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            out = (out + 1) * 127.5
            out = np.clip(out, 0, 255).astype(np.uint8)
            return out
