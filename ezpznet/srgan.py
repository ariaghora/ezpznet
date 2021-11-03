import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
from PIL import Image
from torch import Tensor

from ezpznet import EzPzNet, download_weight

__all__ = ["ResidualConvBlock", "Discriminator", "Generator", "ContentLoss"]


URL_WEIGHT = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131935&authkey=AEQtZdTphjwIyng"


def load_image(image_path: str):
    image = np.array(Image.open(image_path))
    image = (image / 127.5) - 1.0
    image = image.transpose(2, 0, 1).astype(np.float32)[None, ...]
    return image


class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size) // 2,
            bias=True,
        )

        self.weight.data = torch.normal(
            torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02
        )
        self.bias.data = torch.zeros((out_channels))

        for p in self.parameters():
            p.requires_grad = True


class conv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        BN=False,
        act=None,
        stride=1,
        bias=True,
    ):
        super(conv, self).__init__()
        m = []
        m.append(
            _conv(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size) // 2,
                bias=True,
            )
        )

        if BN:
            m.append(nn.BatchNorm2d(num_features=out_channel))

        if act is not None:
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act=nn.ReLU(inplace=True), bias=True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN=True, act=act))
        m.append(conv(channels, channels, kernel_size, BN=True, act=None))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_res_block,
        act=nn.ReLU(inplace=True),
    ):
        super(BasicBlock, self).__init__()
        m = []

        self.conv = conv(in_channels, out_channels, kernel_size, BN=False, act=act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))

        m.append(conv(out_channels, out_channels, kernel_size, BN=True, act=None))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res

        return out


class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act=nn.ReLU(inplace=True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))

        if act is not None:
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out


class discrim_block(nn.Module):
    def __init__(
        self, in_feats, out_feats, kernel_size, act=nn.LeakyReLU(inplace=True)
    ):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN=True, act=act))
        m.append(conv(out_feats, out_feats, kernel_size, BN=True, act=act, stride=2))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        img_feat=3,
        n_feats=64,
        kernel_size=3,
        num_block=16,
        act=nn.PReLU(),
        scale=4,
    ):
        super(Generator, self).__init__()

        self.conv01 = conv(
            in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act
        )

        resblocks = [
            ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)
        ]
        self.body = nn.Sequential(*resblocks)

        self.conv02 = conv(
            in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=True, act=None
        )

        if scale == 4:
            upsample_blocks = [
                Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act)
                for _ in range(2)
            ]
        else:
            upsample_blocks = [
                Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)
            ]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(
            in_channel=n_feats,
            out_channel=img_feat,
            kernel_size=3,
            BN=False,
            act=nn.Tanh(),
        )

    def forward(self, X):
        X = self.conv01(X)
        _skip_connection = X

        X = self.body(X)
        X = self.conv02(X)
        feat = X + _skip_connection

        X = self.tail(feat)
        X = self.last_conv(X)

        return X, feat


class SRGAN(EzPzNet):
    def __init__(self, res_num: int = 16):
        super().__init__()
        self.res_num = res_num

        self.net = Generator(
            img_feat=3, n_feats=64, kernel_size=3, num_block=self.res_num
        )
        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        weights_name = os.path.join(weights_dir, "srgan.pt")
        download_weight(URL_WEIGHT, weights_name)
        self.net.load_state_dict(torch.load(weights_name, map_location=self.device))

    def predict(self, X):
        with torch.no_grad():
            pred, _ = self.net(X)
            return pred
