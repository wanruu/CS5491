import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
from torch.utils.data.dataset import Dataset


PATH = "../../CUB_200_2011/CUB_200_2011/"


class MyDataset(Dataset):
    """
    MyDataset(train=True) for training data
    MyDataset(train=False) for testing data
    """
    def __init__(self, train):
        is_train_df = pd.read_csv(PATH + "train_test_split.txt", sep=" ", header=None, names=["idx", "is_train"])
        paths_df = pd.read_csv(PATH + "images.txt", sep=" ", header=None, names=["idx", "path"])
        total_df = pd.merge(is_train_df, paths_df)
        self.df = total_df[total_df.is_train==int(train)]

    def __getitem__(self, idx):
        img_path = PATH + "images/" + self.df.iloc[idx, 2]
        img = Image.open(img_path).resize((384, 384)).convert("RGB")  # read image and resize
        img_tensor = torch.Tensor(np.array(img))  # convert image to tensor
        img_tensor = rearrange(img_tensor, "w h c -> c w h")  # reorder
        label = int(img_path.split("/")[-2].split(".")[0])  # extract label from image path
        label_tensor = torch.LongTensor([label])  # convert label to tensor
        return img_tensor, label_tensor

    def __len__(self):
        return self.df.shape[0]

