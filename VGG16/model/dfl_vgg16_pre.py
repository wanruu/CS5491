import torch.nn as nn
import torchvision

    
class DFL_VGG16_Pre(nn.Module):
    def __init__(self, class_num, k):
        """
        params class_num: M
        params k        : filters/class
        """
        super().__init__()
        self.name = "DFL_VGG16_Pretrained"
        self.class_num = class_num
        self.k = k
        
        # pretrained
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features

        # Basic feature extraction with vgg
        self.conv1_4 = nn.Sequential(*list(vgg16featuremap.children())[:-11])

        # G-Stream: left, conv5->fc(s)->loss
        self.layer5 = self.layer5 = nn.Sequential(*list(vgg16featuremap.children())[-11:])
        self.g_fc_layers = nn.Sequential(
            nn.Conv2d(512, class_num, kernel_size=1, stride=1),
            nn.BatchNorm2d(class_num),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # P-Stream: mid
        self.conv6 = conv6 = nn.Conv2d(512, k*class_num, kernel_size=1, stride=1)  # filters grouped into {class_num}, out = kM*H*W
        self.pool6 = nn.MaxPool2d(kernel_size=24, stride=24)  # out = kM*1*1 
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
        #print(g.shape)

        # P-stream
        p = self.conv6(conv1_4)  # [32, 2000, 24, 24]
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

    dfl_vgg16_pre = DFL_VGG16_Pre(class_num=CLASS_NUM, k=10)


