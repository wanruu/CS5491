import torch.nn as nn

class DFL_VGG16(nn.Module):
    def __init__(self, class_num, k, vgg, conv6_weight=None, conv6_bias=None):
        """
        params class_num: M
        params k        : filters/class
        """
        super().__init__()
        self.name = "DFL_VGG16"
        if conv6_bias is not None or conv6_weight is not None:
            self.name += "_Unrandom"
        self.class_num = class_num
        self.k = k

        # Basic feature extraction with vgg
        self.conv1_4 = nn.Sequential(vgg.layer1, vgg.layer2, vgg.layer3, vgg.layer4)

        # G-Stream: left, conv5->fc(s)->loss
        self.layer5 = vgg.layer5
        conv5_chann_out = vgg.conv_paras[12][1]
        self.g_fc_layers = nn.Sequential(
            nn.Conv2d(conv5_chann_out, class_num, kernel_size=1, stride=1),
            nn.BatchNorm2d(class_num),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # P-Stream: mid
        conv4_chann_out = vgg.conv_paras[9][1]
        self.conv6 = conv6 = nn.Conv2d(conv4_chann_out, k*class_num, kernel_size=1, stride=1)  # filters grouped into {class_num}, out = kM*H*W
        if conv6_weight is not None:
            self.conv6.weight.data = conv6_weight
        if conv6_bias is not None:
            self.conv6.bias.data = conv6_bias
        self.pool6 = nn.MaxPool2d(kernel_size=12, stride=12)  # out = kM*1*1
        self.p_fc_layers = nn.Sequential(
            nn.Conv2d(k*class_num, class_num, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        # Side-branch: kM x 1 x 1 -> M x 1 x 1
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k, stride=k)


    def forward(self, x):
        batch_size = x.size(0)  # 32

        # Feature extraction
        conv1_4 = self.conv1_4(x)  # [32, 512, 12, 12]

        # G-stream
        g = self.layer5(conv1_4)
        g = self.g_fc_layers(g)
        g = g.view(batch_size, -1)  # [32, 200]

        # P-stream
        p = self.conv6(conv1_4)  # [32, 2000, 12, 12]
        p_inter = self.pool6(p)  # [32, 2000, 1, 1]
        p = self.p_fc_layers(p_inter)
        p = p.view(batch_size, -1)

        # Side-branch
        side = p_inter.view(batch_size, -1, self.k*self.class_num)
        side = self.cross_channel_pool(side)
        side = side.view(batch_size, -1)  # [32, 200]
        return g, p, side


# For testing only.
if __name__ == "__main__":
    from config import *
    from vgg16 import VGG16

    vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
    dfl_vgg16 = DFL_VGG16(class_num=CLASS_NUM, k=K, vgg=vgg16)


