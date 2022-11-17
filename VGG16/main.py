import torch
import torch.nn as nn
from model import VGG16
from train import train
from test import test
from data import MyDataset, AUGMENT

# ---------- #
# Parameters #
# ---------- #
# Data
RESIZE = 192
DATA_PATH = "../../CUB_200_2011/CUB_200_2011/"

# Model
CLASS_NUM = 200
CONV = [
    (3, 64, 3, 1), (64, 64, 3, 1),
    (64, 128, 3, 1), (128, 128, 3, 1),
    (128, 256, 3, 1), (256, 256, 3, 1), (256, 256, 3, 1),
    (256, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
    (512, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
]
MAX_POOL = [(2, 2) for _ in range(5)]  # 2^5 = 32
tmp = int(RESIZE/32)
FC = [(tmp * tmp * 512, 4096), (4096, 4096)]
DROPOUT = 0.5
MODEL_SAVE_PATH = "checkpoint/aug_attempt1"
MODEL_SAVE_INTERVALS = 5

# Training & Testing
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
GPU = torch.cuda.is_available()

print(f"RESIZE={RESIZE}, DATA_PATH={DATA_PATH}, ", end="")
print(f"CLASS_NUM={CLASS_NUM}, CONV={CONV}, MAX_POOL={MAX_POOL}, FC={FC}, DROPOUT={DROPOUT}, MODEL_SAVE_PATH={MODEL_SAVE_PATH}, MODEL_SAVE_INTERVALS={MODEL_SAVE_INTERVALS}")
print(f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, GPU={GPU}")


# --------- #
# Init data #
# --------- #
print(f"Initializing dataset...")
train_data = MyDataset(train=True, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=AUGMENT)
test_data = MyDataset(train=False, img_shape=(RESIZE, RESIZE), path=DATA_PATH, augments=[])

# ---------------- #
# Initialize model #
# ---------------- #
print("Initializing VGG16...")
vgg16 = VGG16(CLASS_NUM, CONV, MAX_POOL, FC, DROPOUT)

# -------- #
# Training #
# -------- #
print("Start training...")
train(vgg16, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, loss_func=None, optimizer=None, early_stopping=None, use_gpu=GPU, save_path=MODEL_SAVE_PATH, save_intervals=MODEL_SAVE_INTERVALS)

# ------- #
# Testing #
# ------- #
print("Start testing...")
test(MODEL_SAVE_PATH + "VGG16.pkl", test_data, batch_size=BATCH_SIZE, use_gpu=GPU)