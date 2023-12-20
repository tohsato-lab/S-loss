import numpy as np
import torch
import torch.nn as nn

import kornia
import kornia.augmentation as K
from torchvision.transforms import functional


class SimTransform(nn.Module):

    def __init__(self):
        super(SimTransform, self).__init__()

        self.transform = nn.Sequential(
            K.RandomResizedCrop((512, 512), scale=(0.2, 1.0), p=1),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(
                mean=torch.Tensor([0.485, 0.456, 0.406]),
                std=torch.Tensor([0.229, 0.224, 0.225])
            )
        )

    def forward(self, img):

        q = self.transform(img)
        k = self.transform(img)

        return [q, k]

class SegTransform(nn.Module):
    def __init__(self, mode):
        super(SegTransform, self).__init__()
        self.mode = mode

        self.k1 = K.RandomHorizontalFlip(p=0.5)
        self.k2 = K.RandomErasing(p=0.1)

        self.k4 = K.RandomRotation(degrees=35.0)

        self.k3_alpha = K.Normalize(
            mean=torch.Tensor([0.485, 0.456, 0.406]),
            std=torch.Tensor([0.229, 0.224, 0.225])
        )

    def forward(self, img, mask):
        if self.mode == 'train':
            img = self.k2(self.k1(img))
            img = self.k4(img)

            mask = self.k1(mask, self.k1._params)
            mask = self.k2(mask, self.k2._params)
            mask = self.k4(mask, self.k4._params)
            mask = mask.long()
        elif self.mode == 'pseudo':
            img = self.k1(img)
            img = self.k2(img)
            img = self.k4(img)
            mask = 0
        elif self.mode == 'test':
            mask = mask.long()

        img = self.k3_alpha(img)
        return img, mask

class ImageNorm(nn.Module):
    def __init__(self):
        super(ImageNorm, self).__init__()
        self.Norm = K.Normalize(
            mean=torch.Tensor([0.485, 0.456, 0.406]),
            std=torch.Tensor([0.229, 0.224, 0.225])
        )
    def forward(self, img, mask):
        img = self.Norm(img)
        if mask != 0:
            mask = mask.long()
        else:
            mask = 0
        return img, mask