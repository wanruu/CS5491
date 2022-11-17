import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL.Image import Image

from utils import box
from einops import rearrange

from configuration import *
import torch
from torch.utils.data.dataset import Dataset
import cv2 as cv
from torch.utils.data import DataLoader


class Raw_CUB_200_2011(Dataset):

    def __init__(self, dataSetName, train=True, path=DataPath, shape=(384, 384), Raw=True):
        super(Raw_CUB_200_2011, self).__init__()
        self.dataSetName = dataSetName
        self._FOR_TRAIN = train
        self._shape = shape
        self.Raw = Raw
        self.path = path
        self.data_path = []
        self.data_label = []
        self._Init()

    def _Init(self):
        target = '1' if self._FOR_TRAIN else '0'
        data_index = []
        with open(self.path + '/train_test_split.txt', 'r') as F:
            for i in F.readlines():
                a = i.split()
                if a[1] == target:
                    data_index.append(a[0])

        with open(self.path + '/images.txt', 'r') as F:
            for i in F.readlines():
                a = i.split()
                if a[0] in data_index:
                    _ = a[1].split('.')
                    label = int(_[0])
                    self.data_label.append(label)
                    self.data_path.append(a[1])

    def __getitem__(self, item):
        if self.Raw:
            img = imageio.imread(self.path + '/images/' + self.data_path[item])
            img = cv.resize(img, self._shape)
            data = torch.Tensor(img)
            data = rearrange(data, "w h c -> c w h")
        else:
            img = imageio.imread(self.path + '/images/' + self.data_path[item])
            mask = imageio.imread(MaskPath + '/' + self.data_path[item][:-3] + 'png')
            img = box(img, mask)
            img = cv.resize(img, self._shape)
            data = torch.Tensor(img)
            data = rearrange(data, "w h c -> c w h")
        return data, torch.LongTensor([self.data_label[item] - 1])

    def _build_new_dataset(self, shape):
        for i in range(len(self.data_path)):
            img = imageio.imread(self.path + '/images/' + self.data_path[i])
            mask = imageio.imread(MaskPath + '/' + self.data_path[i][:-3] + 'png')
            img = box(img, mask)
            img = cv.resize(img, shape)
            cv.imwrite(self.path + '/images/' + self.data_path[i], img)
            if i % 200 == 0:
                print(i, len(self.data_path))

    def __len__(self):
        return len(self.data_label)


class CUB_200_2011(Dataset):
    def __init__(self, dataSetName, train=True, path=DataPath, shape=192):
        super(CUB_200_2011, self).__init__()
        self.dataSetName = dataSetName
        self._FOR_TRAIN = train
        self._shape = shape
        self.path = path
        self.data_path = []
        self.data_label = []
        self._Init()

    def _Init(self):
        target = '1' if self._FOR_TRAIN else '0'
        data_index = []
        with open(self.path + '/train_test_split.txt', 'r') as F:
            for i in F.readlines():
                a = i.split()
                if a[1] == target:
                    data_index.append(a[0])

        with open(self.path + '/images.txt', 'r') as F:
            for i in F.readlines():
                a = i.split()
                if a[0] in data_index:
                    _ = a[1].split('.')
                    label = int(_[0])
                    self.data_label.append(label)
                    self.data_path.append(a[1])

    def __getitem__(self, item):

        img = cv.imread(self.path + '/images/' + self.data_path[item])
        data = torch.Tensor(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    print(img[i, j, k], data[i][j][k])

        data = rearrange(data, "w h c -> c w h")

        label = int(self.path.split("/")[-2].split(".")[0])

        return data, torch.LongTensor([label - 1])

    def __len__(self):
        return len(self.data_label)


if __name__ == '__main__':
    cub_train = CUB_200_2011(1)
    cub_test = CUB_200_2011(0)

    # img = np.array(cub_train[0][0])

    # img =
    # img = np.array(img)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # cv.imshow('ok', img)
    # cv.waitKey(0)

    train_data = DataLoader(cub_train, batch_size=1)

    # for i in range(len(cub_train)[300:303]):
    #     data, label = cub_train[i]
    #
    # for data, label in train_data:
    #     print(label)
    #     # data = rearrange(data[0], 'c w h -> w h c')
    #     print(data.shape)
    #     data = np.array(data[0])
    #     # print(data)
    #     cv.imshow('okokokok', np.uint8(data))
    #     cv.waitKey(0)

    # cub_train._build_new_dataset(shape=(192, 192))
    # cub_test._build_new_dataset(shape=(192, 192))
    # data, label = cub[0]
    # cub._build_new_dataset(shape=(192, 192))
    # print(data.shape)
    # cub.fuck(0)
    # print(cub.data_path[0])
