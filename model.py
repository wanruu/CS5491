import torch.nn as nn
from torchvision import models
from utils import weight_init_kaiming


class Testmodel(nn.Module):

    def __init__(self, class_num, dropout = 0.1):
        super(Testmodel, self).__init__()
        self.Flatten = nn.Flatten()
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 8, (3, 3), stride=(3, 3)), nn.MaxPool2d(2, 2))
        self.cnn2 = nn.Sequential(nn.Conv2d(8, 16, (3, 3), stride=(3, 3)), nn.Dropout(dropout))
        self.cnn3 = nn.Sequential(nn.Conv2d(16, 16, (3, 3), stride=(1, 1)), nn.BatchNorm2d(16))
        self.cnn4 = nn.Sequential(nn.Conv2d(16, 4, (3, 3), stride=(1,1)), nn.Dropout(dropout))
        self.classify = nn.Sequential(nn.Linear(144, class_num+1), nn.Softmax())

    def forward(self, x):
        # x = self.Flatten(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.Flatten(x)
        x = self.classify(x)
        return x


class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*4, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 192, 192)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)

