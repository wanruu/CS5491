import torch
import torch.nn as nn
from model import VGG16
from train import train
from test import test
from data import MyDataset

LEARNING_RATE = 0.01
BATCH_SIZE = 64
CLASS_NUM = 200
DATA_PATH = "../../CUB_200_2011/CUB_200_2011/"

# Load data
print("Initializing dataset...")
train_data = MyDataset(train=True, img_shape=(192, 192), path=DATA_PATH)
test_data = MyDataset(train=False, img_shape=(192, 192), path=DATA_PATH)


# Set hyper parameters
conv_paras = [
    (3, 64, 3, 1), (64, 64, 3, 1),
    (64, 128, 3, 1), (128, 128, 3, 1),
    (128, 256, 3, 1), (256, 256, 3, 1), (256, 256, 3, 1),
    (256, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
    (512, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
]
pool_paras = [(2, 2) for _ in range(5)]  # 2^5 = 32
s = int(192 / 32)
fc_paras = [(s * s * 512, 4096), (4096, 4096)]


# Initialize model
vgg16 = VGG16(CLASS_NUM, conv_paras, pool_paras, fc_paras)


# Training
use_gpu = torch.cuda.is_available()
train(vgg16, train_data, epochs=300, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, loss_func=None, optimizer=None, early_stopping=None, use_gpu=True, save_path="checkpoint/", save_intervals=50)

# Testing
test("checkpoint/VGG16.pkl", test_data, batch_size=BATCH_SIZE, use_gpu=True)