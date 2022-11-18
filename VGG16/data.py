import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    """Dataset for CUB200"""
    def __init__(self, train, img_shape, path, augments=[]):
        """
        params train    : True/False, whether to load training data.
        params img_shape: tuple, to resize image. e.g., (192, 192)
        params augments : function list. [] - no augmentation
        """
        self.img_shape = img_shape
        self.path = path
        self.augments = [[lambda x: x]] + augments

        # Read basic information about dataset.
        is_train_df = pd.read_csv(self.path + "train_test_split.txt", sep=" ", header=None, names=["idx", "is_train"])
        paths_df = pd.read_csv(self.path + "images.txt", sep=" ", header=None, names=["idx", "path"])
        total_df = pd.merge(is_train_df, paths_df)
        self.df = total_df[total_df.is_train==int(train)]

    def __getitem__(self, idx):
        img_idx = int(idx / len(self.augments))
        augment_idx = idx % len(self.augments)
        # Read image data from file
        img_path = self.path + "images/" + self.df.iloc[img_idx, 2]
        img = Image.open(img_path).convert("RGB")
        # Image augmentation
        for func in self.augments[augment_idx]:
            img = func(img)
        # Resize
        img = img.resize(self.img_shape)
        # Convert image to tensor, reorder.
        img_tensor = torch.Tensor(np.array(img))
        img_tensor = rearrange(img_tensor, "w h c -> c w h")
        # Extract label from image path.
        label = int(img_path.split("/")[-2].split(".")[0])
        # Convert label to tensor.
        label_tensor = torch.LongTensor([label])
        return img_tensor, label_tensor

    def __len__(self):
        return self.df.shape[0] * len(self.augments)


def img_rotate(angle: float):
    def inner(img: Image):
        return img.rotate(angle)
    return inner


def img_flip():
    def inner(img: Image):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return inner


AUGMENT = [[img_rotate(15)], [img_rotate(-15)], [img_rotate(15), img_flip()], [img_rotate(-15), img_flip()]]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import *

    train_data = MyDataset(train=True, img_shape=(384, 384), path=DATA_PATH, augments=AUGMENT)
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12)

