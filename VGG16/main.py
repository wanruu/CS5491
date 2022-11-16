from model import VGG16
from train import train
from data import MyDataset
from torch.utils.data import DataLoader


BATCH_SIZE = 64
CLASS_NUM = 200

# load data
print("Loading data...")
train_data = MyDataset(train=True, img_shape=(192, 192))
test_data = MyDataset(train=False, img_shape=(192, 192))


# hyper parameters
conv_paras = [
    (3, 64, 3, 1), (64, 64, 3, 1),
    (64, 128, 3, 1), (128, 128, 3, 1),
    (128, 256, 3, 1), (256, 256, 3, 1), (256, 256, 3, 1),
    (256, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
    (512, 512, 3, 1), (512, 512, 3, 1), (512, 512, 3, 1),
]
pool_paras = [(2, 2) for _ in range(5)]  # 2^5 = 32
s = int(192 / 32)
fc_paras = [(s * s * 512, 1024), (1024, 1024)]


vgg16 = VGG16(CLASS_NUM, conv_paras, pool_paras, fc_paras)
train(vgg16, train_data, epochs=1, batch_size=BATCH_SIZE, learning_rate=0.01, loss_func=None, optimizer=None)


