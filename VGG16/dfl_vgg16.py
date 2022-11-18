import torch
import torch.nn as nn
from model import VGG16, DFL_VGG16
from train import train
from test import test
from data import MyDataset, AUGMENT
from config import *


VGG16_PATH = "VGG16-epoch=6.pt"


# --------- #
# Init data #
# --------- #
print(f"Initializing dataset...")
train_data = MyDataset(train=True, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=AUGMENT)
test_data = MyDataset(train=False, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=[])

# ---------------- #
# Initialize model #
# ---------------- #
print("Initializing DFL-VGG16...")
vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
vgg16.load_state_dict(torch.load(VGG16_PATH))
# for param_tensor in vgg16.state_dict():
#     print(param_tensor, "\t", vgg16.state_dict()[param_tensor].size())
dfl_vgg16 = DFL_VGG16(class_num=CLASS_NUM, k=K, vgg=vgg16, img_shape=(RESIZE, RESIZE))


# -------- #
# Training #
# -------- #
print("Start training...")
dfl_vgg16 = train(dfl_vgg16, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, loss_func=None, optimizer=None, early_stopping=None, use_gpu=GPU, save_path=MODEL_SAVE_PATH, save_intervals=MODEL_SAVE_INTERVALS)

# # ------- #
# # Testing #
# # ------- #
# print("Start testing...")
# test(dfl_vgg16, test_data, batch_size=BATCH_SIZE, use_gpu=GPU)