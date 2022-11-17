import imageio.v2 as imageio
import numpy as np
from utils import box
from einops import rearrange

from configuration import *
import torch
from torch.utils.data.dataset import Dataset
import cv2 as cv


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
    def __init__(self, dataSetName, train=True, path=DataPath, shape=(192, 192)):
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
        img = imageio.imread(self.path + '/images/' + self.data_path[item])
        data = torch.Tensor(img)
        data = rearrange(data, "w h c -> c w h")
        return data, torch.LongTensor([self.data_label[item] - 1])

    def __len__(self):
        return len(self.data_label)


class R(Dataset):
    def __init__(self):
        super(R, self).__init__()
        self._list = [0] * 1000

    def __getitem__(self, item):
        data = torch.randn((3, 384, 384))
        label = torch.tensor([0])
        return data, label

    def __len__(self):
        return len(self._list)


if __name__ == '__main__':
    # cub_train = Raw_CUB_200_2011(dataSetName='cub', train=True)
    # cub_test = Raw_CUB_200_2011(dataSetName='cub', train=False)
    r = R()
    data, label = r[0]
    print(data.shape, label.shape)

    # cub_train._build_new_dataset(shape=(192, 192))
    # cub_test._build_new_dataset(shape=(192, 192))
    # data, label = cub[0]
    # cub._build_new_dataset(shape=(192, 192))
    # print(data.shape)
    # cub.fuck(0)
    # print(cub.data_path[0])
