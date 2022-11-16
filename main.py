from model import VGG16, Testmodel
from train import train
from test import test
from utils import count_parameters, Log, get_device
from evaluate import evalMatrix
from data import CUB_200_2011
from configuration import *

# hyper parameters
conv_paras = [
    (3, 64, 3, 1), (64, 64, 3, 1),
    (64, 128, 3, 1), (128, 128, 3, 1),
    (128, 256, 3, 1), (256, 256, 3, 1), (256, 256, 3, 1),
    (256, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
    (512, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
]
pool_paras = [(2, 2) for _ in range(5)]  # 2^5 = 32
s = int(384 / 32)
fc_paras = [(s * s * 512, 4096), (4096, 4096)]


def main():
    train_data = CUB_200_2011('cub', train=True)
    test_data = CUB_200_2011('cub', train=False)
    device = get_device()
    train_evaluator = evalMatrix(clses=classNumber, device=device)
    test_evaluator = evalMatrix(clses=classNumber, device=device)
    log = Log(LogPath + 'record.csv', model_name='cnn')

    # -----------------

    # vgg16 = VGG16(8, conv_paras, pool_paras, fc_paras)
    testmodel = Testmodel()

    train(testmodel, train_data, evaluator=train_evaluator, log=log)
    test(model=testmodel, dataset=test_data, device=device, batch_size=64, test_evaluater=test_evaluator, log=log)

    # -----------------


if __name__ == '__main__':
    main()
