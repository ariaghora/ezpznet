import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ..ezpznet import EzPzNet, download_weight


URL_WEIGHT = "https://onedrive.live.com/download?cid=5D199F9D83F53944&resid=5D199F9D83F53944%2131927&authkey=AP3wa4xmfYsocDU"


def get_model():
    model = nn.Sequential(  # Sequential,
        nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
        nn.ReLU(),
        nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
        nn.Sigmoid(),
    )
    return model


class SketchGAN(EzPzNet):
    def __init__(self):
        self.model = get_model()

        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        weights_name = os.path.join(weights_dir, "sketchgan.pt")
        download_weight(URL_WEIGHT, weights_name)
        self.model.load_state_dict(torch.load(weights_name))

    def predict(self, X):
        with torch.no_grad():
            return self.model(X)


def load_image(image_path: str):
    immean = 0.9664114577640158
    imstd = 0.0858381272736797
    data = Image.open(image_path).convert("L")
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
    return data
