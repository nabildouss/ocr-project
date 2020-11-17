import torch
from torch import nn


class Reshape(nn.Module):

    def __init__(self, s_out):
        super().__init__()
        self.s_out = s_out

    def forward(self, x):
        return torch.reshape(x, (x.shape[0],*self.s_out))


class Transpose(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.transpose(*self.args)


class ColumnPooling(nn.Module):

    def __init__(self, shape_in, n_columns=None):
        super().__init__()
        self.shape_in = shape_in
        k_height = int(shape_in[0])
        k_width = int(shape_in[1])
        self.n_columns = 1 #n_columns
        self.pooling = nn.MaxPool2d(kernel_size=(k_height, 1), stride=1)

    def forward(self, f_maps):
        y = self.pooling(f_maps)
        return y.reshape(f_maps.shape[0], f_maps.shape[1]*f_maps.shape[3])


class BaseLine(nn.Module):

    def __init__(self, shape_in=(1, 64, 512), n_pool_columns=100, n_char_class=100, sequence_length=100):
        super().__init__()
        in_channels, h, w =  shape_in
        conv_layer = lambda c_in, c_out: nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=c_in, out_channels=c_out,
                                                                 padding=1), nn.ReLU(), nn.BatchNorm2d(c_out))
        self.conv1 = nn.Sequential(conv_layer(in_channels, 64), conv_layer(64, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(conv_layer(64, 128), conv_layer(128, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(conv_layer(128, 256), conv_layer(256, 256),
                                   conv_layer(256, 512), conv_layer(512, 512))
        f_scale = 1/self.pool1.stride/self.pool2.stride
        self.pool3 = ColumnPooling(shape_in=(h*f_scale, w*f_scale), n_columns=n_pool_columns)
        fc_layer = lambda c_in, c_out: nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out), nn.ReLU())
        featuer_in = shape_in[2]*f_scale*512
        self.fc1 = nn.Sequential(fc_layer(int(featuer_in), 1024), fc_layer(1024, 1024), nn.Dropout(p=0.5),
                                 fc_layer(1024, 1024), nn.Dropout(p=0.5))
        self.out = nn.Sequential(fc_layer(1024, 1024), nn.Linear(1024, n_char_class * sequence_length),
                                 Reshape([sequence_length, n_char_class]), nn.LogSoftmax(dim=2), Transpose([0, 1]))

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.pool3(y)
        y = self.fc1(y)
        return self.out(y)
