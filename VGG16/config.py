import os
import torch
import datetime


# Data
RESIZE = 192
DATA_PATH = "../../CUB_200_2011/CUB_200_2011/"

# Model
CLASS_NUM = 200
CONV = [
    (3, 64), (64, 64),
    (64, 128), (128, 128),
    (128, 256), (256, 256), (256, 256),
    (256, 512), (512, 512), (512, 512),
    (512, 512), (512, 512), (512, 512),
]
tmp = int(RESIZE/32)
FC = [(tmp * tmp * 512, 4096), (4096, 4096)]
DROPOUT = 0.5
MODEL_SAVE_PATH = f"checkpoint/{datetime.datetime.now()}"
MODEL_SAVE_INTERVALS = 5

K = 10

# Training & Testing
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.01
GPU = torch.cuda.is_available()



if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

print(f"RESIZE={RESIZE}")
print(f"CLASS_NUM={CLASS_NUM}")
print(f"CONV={CONV}")
print(f"FC={FC}")
print(f"DROPOUT={DROPOUT}")
print(f"EPOCHS={EPOCHS}")
print(f"BATCH_SIZE={BATCH_SIZE}")
print(f"LEARNING_RATE={LEARNING_RATE}")
print(f"GPU={GPU}")
print(f"K={K}")
print()