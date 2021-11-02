from abc import ABC, abstractmethod
from typing import Any

import requests
import os
from tqdm import tqdm


def download_weight(url: str, fname: str):
    if not os.path.isfile(fname):
        # attempt to create dir
        dir_name = os.path.dirname(fname)
        os.makedirs(dir_name, exist_ok=True)

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)


class EzPzNet(ABC):
    def __init__(self):
        self.device = "cpu"

    @abstractmethod
    def predict(self, X: Any):
        raise NotImplementedError()
