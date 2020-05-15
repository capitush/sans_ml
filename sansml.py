import os
from PIL import Image
import numpy as np
import torch
import cv2


class DataSet(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # there is only one class
        label = ""

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        return len(self.imgs)
