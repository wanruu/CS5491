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


# For testing only.
if __name__ == "__main__":
    from config import *
    vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)


