import torch
from torch import nn
from src.sliding_window import SlidingWindow, AutomaticWindow


class Reshape(nn.Module):
    """
    This class adapts the reshape function to the nn.Module duck-typing. Reshape can be used like a layer.
    """

    def __init__(self, s_out):
        super().__init__()
        self.s_out = s_out

    def forward(self, x):
        return torch.reshape(x, (x.shape[0],*self.s_out))


class Transpose(nn.Module):
    """
    This class adapts the transpose function to the nn.Module duck-typing. Transpose can be used like a layer.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.transpose(*self.args)


class Permute(nn.Module):

    def __init__(self, idcs):
        super().__init__()
        self.idcs = idcs

    def forward(self, x):
        return x.permute(*self.idcs)


class ColumnPooling(nn.Module):
    """
    This class pools columns and columns only, allowing us to extract more information with respect to time.
    """

    def __init__(self, shape_in, stride=4):
        super().__init__()
        self.shape_in = shape_in
        k_height = int(shape_in[0])
        #k_width = int(shape_in[1])
        self.pooling = nn.MaxPool2d(kernel_size=(k_height, stride), stride=stride)
        self.stride = stride
        self.f_scale = 1 / (stride)

    def forward(self, f_maps):
        y = self.pooling(f_maps)
        return y
        #return y.reshape(y.shape[0], y.shape[1]*y.shape[3])


class BaseLine(nn.Module):
    """
    This class defines a CNN, used to provide the input for the CTCLoss function (i.e. the probabilities).

    The CNN consists of 3 pooling phases, two regular max pooling phases and one final pooling phase, that only
    considers the columns, i.e. pooling with respect to time in the context of images.
    The CNN features Batch-Normalization for regularization.

    A fully connected Classifier network is concatenated to the back of the CNN, generating the probabilities.
    The fully connected network features dropout for regularization.
    The output is reshaped and transposed to fit PyTorch's CTCLoss function.
    """

    def __init__(self, shape_in=(1, 32, 3*32), n_char_class=100, sequence_length=256, dropout=0.1):
        """
        :param shape_in: shape of the input images
        :param n_char_class: number of character classes (required as we calculate the prob. for CTC)
        :param sequence_length: maximum length of a sequence(/ line)
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.n_char_class = n_char_class
        # model used to estimate log-pobs of a sequence step
        self.cnn = CharHistCNN(shape_in, n_char_class, sequence_length, dropout)#CNN(shape_in, n_char_class, sequence_length, dropout)
        self.permute = Permute([0, 3, 1, 2])
        #self.dense = nn.Sequential(nn.Linear(self.cnn.d_out[0] * self.cnn.d_out[1], n_char_class), nn.ReLU())
        self.sliding_window = SlidingWindow(seq_len=self.sequence_length)
        self.lstm1 = nn.LSTM(input_size=int(n_char_class), hidden_size=n_char_class, num_layers=3, #dropout=0.25,#)
                             bidirectional=False, batch_first=True)
        #self.fc = nn.Sequential(nn.Linear(2*n_char_class, n_char_class))
        #self.fc = nn.Sequential(nn.Linear(n_char_class, n_char_class))

    def forward(self, batch):
        s_windows = []
        for img in batch:
            s_windows.append(self.sliding_window.sliding_windows(img))
        s_windows = torch.cat(s_windows)#s_windows)
        y = self.cnn(s_windows)#batch)
        #y = self.permute(y)
        y = y.view(batch.shape[0], self.sequence_length, self.n_char_class)
        #y = self.dense(y)
        y, _ = self.lstm1(y)
        #y = self.fc(y)
        y = y.transpose(1, 0) # T, N, C
        return nn.functional.log_softmax(y[:,:,:self.n_char_class], dim=2)


class CNN(nn.Module):

    def __init__(self, shape_in=(1, 64, 64), n_char_class=100, sequence_length=256, dropout=0.1):
        """
        :param shape_in: shape of the input images
        :param n_char_class: number of character classes (required as we calculate the prob. for CTC)
        :param sequence_length: maximum length of a sequence(/ line)
        """
        # -------------------------------------------model setup overhead-------------------------------------------
        super().__init__()
        # scope of the CTC / character sequences
        self.sequence_length = sequence_length
        self.n_char_class = n_char_class
        # number of image channels (1 channel for BW images)
        in_channels, h, w = shape_in
        # a generic definition of convolution layers, all layers shall have the same activation and batch normalization
        conv_layer = lambda c_in, c_out: nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=c_in, out_channels=c_out,
                                                                 padding=1), 
                                                       nn.ReLU(), 
                                                       nn.BatchNorm2d(c_out))

        # -------------------------------------------the CNN architecture-------------------------------------------
        # first phase: operating on the original image, no more than 64 channels
        self.conv1 = nn.Sequential(conv_layer(in_channels, 8),
                                   conv_layer(8, 8))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))# 64x1024-> 32x512

        self.conv2 = nn.Sequential(conv_layer(8, 16),
                                   conv_layer(16, 16))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))# 64x512-> 32x256

        self.conv3 = nn.Sequential(conv_layer(16, 32),
                                   conv_layer(32, 32))
        #self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))# 32x256 -> 16x256

        #self.conv4 = nn.Sequential(conv_layer(32, 64),
        #                           conv_layer(64, 64))
        #self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))# 16x256 -> 8x256

        #self.conv5 = nn.Sequential(conv_layer(64, 128),
        #                           conv_layer(128, 128))
        #self.pool5 = nn.MaxPool2d(kernel_size=(2, 1))# 8x256 -> 4x256
        out_r = shape_in[1] / 2/2#/2/2/2
        out_c = shape_in[2] / 2/2
        self.d_out = (32, int(out_r), int(out_c))

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        #y = self.pool3(y)

        #y = self.conv4(y)
        #y = self.pool4(y)

        #y = self.conv5(y)
        #y = self.pool5(y)
        return y


class CharHistCNN(nn.Module):


    def __init__(self, shape_in=(1, 32, 3*32), n_char_class=100, sequence_length=256, dropout=0.1):
        super().__init__()
        self.cnn = CNN(shape_in, n_char_class, sequence_length, dropout)
        # a generic definition of fully connected layers, all layers shall have the same activatin
        fc_layer = lambda c_in, c_out: nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out), nn.ReLU())
        # --------------------------------the probability prediction--------------------------------------
        #stride_prod = self.cnn.pool1.stride * self.cnn.pool2.stride
        #if shape_in[1] % stride_prod != 0 or shape_in[2] % stride_prod != 0:
        #    raise ValueError(f'culd not initialize model: image width and height aught to be divisible by {stride_prod}')
        self.fc1 = nn.Sequential(fc_layer(self.cnn.d_out[0] * self.cnn.d_out[1] * self.cnn.d_out[2], n_char_class))#1024),
                                 #fc_layer(1024, 1024), nn.Dropout(0.5),
                                 #fc_layer(1024, 1024), nn.Dropout(0.5))
        #self.out = nn.Sequential(fc_layer(1024, int(n_char_class)))#,
                                 #nn.Linear(1024, n_char_class), nn.LogSoftmax(dim=1))#nn.Sigmoid())#

    def forward(self, x):
        """
        Performs a forward pass of the neural network.

        :param x: batch of images
        :return: probabilities for PyTorch's CTCLoss fucntion
        """
        #print(x.shape)
        # having defined the phases and activations already, the forward pass implementation is quite clean
        y = self.cnn(x)
        #print(y.shape)
        y = torch.flatten(y, start_dim=1)
        #print(y.shape)
        #print(self.cnn.d_out)
        y = self.fc1(y)
        return y#self.out(y)

