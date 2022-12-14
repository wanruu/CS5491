import torch

from model import VGG16, Testmodel, ResNet
from train import train
from test import test
from utils import count_parameters, Log, get_device
from evaluate import evalMatrix
from data import CUB_200_2011
from configuration import *
import torch.nn as nn


# ------------------------------------------------------------
# |                  hyper-parameter pool                    |
# |                                                          |

epochs = 100
lr = 1e-3
loss_func = nn.CrossEntropyLoss()


# |                                                          |
# |                  hyper-parameter pool                    |
# ------------------------------------------------------------


def main():
    train_data = CUB_200_2011('cub', train=True, shape=192)
    test_data = CUB_200_2011('cub', train=False, shape=192)

    # ==============================================

    device = get_device()
    train_evaluator = evalMatrix(clses=classNumber, device=device)
    test_evaluator = evalMatrix(clses=classNumber, device=device)
    log = Log(LogPath + 'record.csv', model_name='cnn')

    # ==============================================

    resnet = ResNet(model_choice=50)
    count_parameters(resnet)

    resnet.load_state_dict(torch.load(SaveModel + 'best_model.pt'))

    train(resnet, train_data, test_data,
          epochs=epochs,
          learning_rate=lr,
          device=device,
          loss_func=loss_func,
          num_workers=2,
          train_evaluator=train_evaluator,
          val_evaluator=test_evaluator,
          log=log)

    # test(model=resnet,
    #      dataset=test_data,
    #      device=device,
    #      batch_size=64,
    #      test_evaluater=test_evaluator,
    #      log=log)

    # -----------------


if __name__ == '__main__':
    main()
