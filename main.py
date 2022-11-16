from model import VGG16
from train import training


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
fc_paras = [(s*s*512, 4096), (4096, 4096)]


vgg16 = VGG16(8, conv_paras, pool_paras, fc_paras)


# TODO: import data
data = []
training(vgg16, data)
