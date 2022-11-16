import torch.nn as nn


def conv_relu_layer(in_chann, out_chann, kernerl_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, kernerl_size, padding), 
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
    def __init__(self, class_num, conv_paras, pool_paras):
        """
        conv_paras: a length-13 list of tuple for convolution layer. Each tuple is (in_chann, out_chann, kernerl_size, padding)
        pool_paras: a length-5 list of tuple for pooling layer. Each tuple is (kernel_size, stride)
        """
        super().__init__()
        
        # Convolution + ReLU layer
        conv_layers = [conv_relu_layer(a, b, c, d) for a, b, c, d in conv_paras]
        # pooling layer
        pool_layers = [nn.MaxPool2d(kernel_size=k_size, stride=stride) for k_size, stride in pool_paras]


        # Full connected layers
        fc_layers = nn.Sequential(
            fc_relu_layer(7*7*512, 4096),
            fc_relu_layer(4096, 4096)
        )

        # Final layer
        final_layer = nn.Linear(4096, class_num)

        # Connect layers
        self.layers = nn.Sequential(
            conv_layers[0], conv_layers[1], pool_layers[0],
            conv_layers[2], conv_layers[3], pool_layers[1],
            conv_layers[4], conv_layers[5], conv_layers[6], pool_layers[2],
            conv_layers[7], conv_layers[8], conv_layers[9], pool_layers[3],
            conv_layers[10], conv_layers[11], conv_layers[12], pool_layers[4],
            fc_layers,
            final_layer
        )

    def forward(self, x):
        return self.layers(x)






