# cutout.py

import numpy as np
import torch


class Cutout(object):
    """
    Randomly masks out one or more patches from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length: int) -> None:
        self.length: int = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask_torch = torch.from_numpy(mask)
        mask_torch = mask_torch.expand_as(img)
        img *= mask_torch
        return img
