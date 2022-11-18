import os
import shutil

import torch
import torch.nn as nn

from model.vgg16 import VGG16
from model.dfl_vgg16 import DFL_VGG16
from model.dfl_vgg16_pre import DFL_VGG16_Pre

from train import train
from test import test
from data import MyDataset
from config import *



def main(model_name):
    print("Initializing dataset...")
    train_data = MyDataset(train=True, path=DATA_PATH, transform=TRAIN_TRANS)
    test_data = MyDataset(train=False, path=DATA_PATH, transform=TEST_TRANS)

    print(f"Initializing {model_name}...")
    if model_name == "VGG16":
        model = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
    elif model_name == "DFL_VGG16":
        vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
        # vgg16.load_state_dict(torch.load("xx.pt"))
        model = DFL_VGG16(class_num=CLASS_NUM, k=K, vgg=vgg16)
    elif model_name == "DFL_VGG16_Pre":
        model = DFL_VGG16_Pre(class_num=CLASS_NUM, k=K)


    print("Start training...")
    model = train(model, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, loss_func=None, optimizer=None, early_stopping=None, use_gpu=GPU, save_path=MODEL_SAVE_PATH, save_intervals=MODEL_SAVE_INTERVALS)

    print("Start testing...")
    test(model, test_data, batch_size=BATCH_SIZE, use_gpu=GPU)



if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

model_name = ["VGG16", "DFL_VGG16", "DFL_VGG16_Pre"][2]
main(model_name)


if not os.listdir(MODEL_SAVE_PATH):
    shutil.rmtree(MODEL_SAVE_PATH)


