import os
import shutil

import torch
import torch.nn as nn

from model.vgg16 import VGG16
from model.dfl_vgg16 import DFL_VGG16
from model.dfl_vgg16_pre import DFL_VGG16_Pre

from train import train
from test import test
from data import MyDataset, AUGMENT
from config import *



def main(model_name, arguments):
    print("Initializing dataset...")
    train_data = MyDataset(train=True, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=arguments)
    test_data = MyDataset(train=False, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=[])

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



model_name = ["VGG16", "DFL_VGG16", "DFL_VGG16_Pre"][0]
augments = AUGMENT
main(model_name, augments)


if not os.listdir(MODEL_SAVE_PATH):
    shutil.rmtree(MODEL_SAVE_PATH)