import torch.nn as nn


def conv_relu_layer(in_chann, out_chann, kernerl_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, kernerl_size, padding=padding), 
        nn.BatchNorm2d(out_chann), 
        nn.ReLU()
    )


def fc_relu_layer(in_chann, out_chann):
    return nn.Sequential(
        nn.Linear(in_chann, out_chann),
        nn.BatchNorm1d(out_chann),
        nn.ReLU()
    )


class VGG16(nn.Module):
    def __init__(self, class_num, conv_paras, fc_paras, dropout=0.5):
        """
        params conv_paras: a length-13 list of tuple for convolution layer. Each tuple is (in_chann, out_chann)
        params fc_paras  : a length-2 list of tuple for fully connection layer. Each tuple is (in_chann, out_chann)
        params dropout   : prevent overfitting
        """
        super().__init__()
        self.name = "VGG16"
        self.class_num = class_num
        self.conv_paras = conv_paras
        self.fc_paras = fc_paras
        self.dropout = dropout
        
        # Convolution + ReLU layer
        conv_layers = [conv_relu_layer(a, b, 3, 1) for a, b in conv_paras]
        # Pooling + ReLU layer
        pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        # Full connected layers
        fc_layers = [fc_relu_layer(a, b) for a, b in fc_paras]
        # Final layer
        final_layer = nn.Linear(fc_paras[-1][1], class_num)

        # Connect layers
        self.layer1 = nn.Sequential(conv_layers[0], conv_layers[1], pool_layer, nn.Dropout(dropout))
        self.layer2 = nn.Sequential(conv_layers[2], conv_layers[3], pool_layer, nn.Dropout(dropout))
        self.layer3 = nn.Sequential(conv_layers[4], conv_layers[5], conv_layers[6], pool_layer, nn.Dropout(dropout))
        self.layer4 = nn.Sequential(conv_layers[7], conv_layers[8], conv_layers[9], pool_layer, nn.Dropout(dropout))
        self.layer5 = nn.Sequential(conv_layers[10], conv_layers[11], conv_layers[12], pool_layer, nn.Dropout(dropout))
        self.layer6 = nn.Sequential(fc_layers[0], fc_layers[1])
        self.layer7 = final_layer


    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = output.view(output.size(0), -1)  # resize
        output = self.layer6(output)
        output = self.layer7(output)
        return output


class DFL_VGG16(nn.Module):
    def __init__(self, class_num, k, vgg, img_shape):
        """
        params class_num: M
        params k        : filters/class
        """
        super().__init__()
        self.name = "DFL_VGG16"
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
        self.pool6 = nn.MaxPool2d(kernel_size=img_shape, stride=img_shape)  # out = kM*1*1
        self.p_fc_layers = nn.Sequential(
            nn.Conv2d(k*class_num, class_num, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        # Side-branch: kM x 1 x 1 -> M x 1 x 1
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k, stride=k)


    def forward(self, x):
        batch_size = x.size(0)

        # Feature extraction
        conv1_4 = self.conv1_4(x)

        # G-stream
        g = self.layer5(conv1_4)
        g = self.g_fc_layers(g)
        g = g.view(batch_size, -1)

        # P-stream
        p = self.conv6(conv1_4)
        p_inter = self.pool6(p)
        p = self.p_fc_layers(p_inter)
        p = p.view(batchsize, -1)

        # Side-branch
        side = p_inter.view(batchsize, -1, self.k*self.class_num)
        side = self.cross_channel_pool(side)
        side = side.view(batchsize, -1)
        print(g.size, p.size, side.size)
        return g, p, side



# For testing only.
if __name__ == "__main__":
    import torch
    
    RESIZE = 192
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
    vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)

    dfl_vgg16 = DFL_VGG16(class_num=CLASS_NUM, k=10, vgg=vgg16, img_shape=(RESIZE, RESIZE))


