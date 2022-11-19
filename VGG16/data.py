import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    """Dataset for CUB200"""
    def __init__(self, train, path, masked=False, transform=None):
        """
        params train    : True/False, whether to load training data.
        params img_shape: tuple, to resize image. e.g., (192, 192)
        params augments : function list. [] - no augmentation
        """
        self.path = path
        self.transform = transform
        self.masked = masked

        # Read basic information about dataset.
        is_train_df = pd.read_csv(self.path + "train_test_split.txt", sep=" ", header=None, names=["idx", "is_train"])
        paths_df = pd.read_csv(self.path + "images.txt", sep=" ", header=None, names=["idx", "path"])
        total_df = pd.merge(is_train_df, paths_df)
        self.df = total_df[total_df.is_train==int(train)]

    def __getitem__(self, idx):
        # Read image data from file
        if self.masked:
            img_path = self.path + "masked_images/" + self.df.iloc[idx, 2]
        else:
            img_path = self.path + "images/" + self.df.iloc[idx, 2]
        
        # Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # Extract label from image path.
        label = int(img_path.split("/")[-2].split(".")[0])
        label = torch.LongTensor([label])

        return img, label

    def __len__(self):
        return self.df.shape[0]



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from config import *


    train_data = MyDataset(train=True, path=DATA_PATH, masked=True, transform=TRAIN_TRANS)
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    for img, label in dataloader:
        print(label)
        x = transforms.ToPILImage()(img[0])
        x.show()
        break

        



