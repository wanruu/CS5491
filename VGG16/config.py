import torch

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
MODEL_SAVE_PATH = "checkpoint/aug_attempt2/"
MODEL_SAVE_INTERVALS = 5

# Training & Testing
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.01
GPU = torch.cuda.is_available()

print(f"RESIZE={RESIZE}, DATA_PATH={DATA_PATH}")
print(f"CLASS_NUM={CLASS_NUM}, CONV={CONV}, FC={FC}, DROPOUT={DROPOUT}")
print(f"MODEL_SAVE_PATH={MODEL_SAVE_PATH}, MODEL_SAVE_INTERVALS={MODEL_SAVE_INTERVALS}")
print(f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, GPU={GPU}")