import imageio.v2 as imageio
from configuration import *
import torch
from torch.utils.data.dataset import Dataset
import cv2 as cv


class CUB_200_2011(Dataset):


    def __init__(self, dataSetName, train=True, path=DataPath, shape=(384, 384)):
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
        img = cv.resize(img, self._shape)
        data = torch.Tensor(img)
        return data, torch.LongTensor(self.data_label[item])

    def __len__(self):
        return len(self.data_label)

    def fuck(self, index):
        data, label = self.__getitem__(index)
        # 60.0 27.0 325.0 304.0
        cv.imshow(data[60:])


if __name__ == '__main__':
    cub = CUB_200_2011(dataSetName='cub')
    data_, label = cub[0]
    cub.fuck(0)
    # print(data.data_path[0])