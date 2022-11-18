import torch
import torch.nn as nn
from model import DFL_VGG16_PRE
from train import train_dfl
from test import test_dfl
from data import MyDataset, AUGMENT
from config import *


VGG16_PATH = "VGG16-epoch=6.pt"
import datetime
import os
MODEL_SAVE_PATH = f"checkpoint/{datetime.datetime.now()}"
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

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
dfl_vgg16 = DFL_VGG16_PRE(class_num=CLASS_NUM, k=K)


# -------- #
# Training #
# -------- #
print("Start training...")
dfl_vgg16 = train_dfl(dfl_vgg16, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, loss_func=None, optimizer=None, early_stopping=None, use_gpu=GPU, save_path=MODEL_SAVE_PATH, save_intervals=MODEL_SAVE_INTERVALS)

# # ------- #
# # Testing #
# # ------- #
print("Start testing...")
test_dfl(dfl_vgg16, test_data, batch_size=BATCH_SIZE, use_gpu=GPU)