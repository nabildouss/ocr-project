import torch
from torch import nn


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

    def __init__(self, shape_in=(1, 64, 512), n_char_class=100, sequence_length=256, dropout=0.1):
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
                                                                 padding=1), nn.ReLU())#, nn.Dropout(dropout))
        # a generic definition of fully connected layers, all layers shall have the same activatin
        fc_layer = lambda c_in, c_out: nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out), nn.ReLU())

        # -------------------------------------------the CNN architecture-------------------------------------------
        # first phase: operating on the original image, no more than 64 channels
        self.conv1 = nn.Sequential(conv_layer(in_channels, 16),
                                   conv_layer(16, 16))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # second phase: column pooling, extracting features w.r.t. time
        self.conv2 = nn.Sequential(conv_layer(16, 16),
                                   conv_layer(16, 16))
        self.pool2 = ColumnPooling(shape_in=(shape_in[1]/self.pool1.stride, shape_in[2]/self.pool1.stride), stride=1)
        # third phase: finally, after have pooled twice, we allow the CNN to be even deeper and have 256 feature maps

        # --------------------------------the probability prediction--------------------------------------
        self.lstm = nn.Sequential(Reshape([16, int(shape_in[2]/self.pool1.stride)]), Transpose([2,1]),
                                  nn.LSTM(input_size=16, hidden_size=n_char_class, num_layers=10, batch_first=True))
        self.out = nn.Sequential(nn.LogSoftmax(dim=2), Transpose([0, 1]))  #Reshape([sequence_length, n_char_class]), 

    def forward(self, x):
        """
        Performs a forward pass of the neural network.

        :param x: batch of images
        :return: probabilities for PyTorch's CTCLoss fucntion
        """
        # having defined the phases and activations already, the forward pass implementation is quite clean
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y, _ = self.lstm(y)
        return self.out(y)
