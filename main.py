from model import VGG16
from train import train
from data import CUB_200_2011

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
fc_paras = [(s * s * 512, 4096), (4096, 4096)]


def main():
    data = CUB_200_2011('cub', train=True)[0:100]
    # for i in range(2568, len(data)):
    #     print(i, data[i][0].shape, data[i][1].shape)

    print(data[748][0].shape, data[748][1].shape)
    print(data[749][0].shape, data[749][1].shape)
    print(data[750][0].shape, data[750][1].shape)
    print(data[2568][0].shape, data[2568][1].shape)
    vgg16 = VGG16(8, conv_paras, pool_paras, fc_paras)
    train(vgg16, data)


if __name__ == '__main__':
    main()
