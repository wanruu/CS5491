import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    """
    MyDataset(train=True) for training data
    MyDataset(train=False) for testing data
    """
    def __init__(self, train, img_shape, path):
        """
        params train    : True/False, whether to load training data.
        params img_shape: tuple, to resize image. e.g., (192, 192)
        """
        self.img_shape = img_shape
        self.path = path
        # Read basic information about dataset.
        is_train_df = pd.read_csv(self.path + "train_test_split.txt", sep=" ", header=None, names=["idx", "is_train"])
        paths_df = pd.read_csv(self.path + "images.txt", sep=" ", header=None, names=["idx", "path"])
        total_df = pd.merge(is_train_df, paths_df)
        self.df = total_df[total_df.is_train==int(train)]

    def __getitem__(self, idx):
        # Read image data from file, resize.
        img_path = self.path + "images/" + self.df.iloc[idx, 2]
        img = Image.open(img_path).resize(self.img_shape).convert("RGB")
        # Convert image to tensor, reorder.
        img_tensor = torch.Tensor(np.array(img))
        img_tensor = rearrange(img_tensor, "w h c -> c w h")
        # Extract label from image path.
        label = int(img_path.split("/")[-2].split(".")[0])
        # Convert label to tensor.
        label_tensor = torch.LongTensor([label])
        return img_tensor, label_tensor

    def __len__(self):
        return self.df.shape[0]
