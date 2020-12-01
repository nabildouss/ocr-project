"""
baseline examples
"""
import torch
from torch import nn
from src.sliding_window import SlidingWindow, AutomaticWindow


class Kraken(nn.Module):


    def __init__(self, n_char_class=262):
        super().__init__()
        self.height = 48
        self.in_channels = 1
        self.n_char_class = n_char_class
        self.sequence_length = 150
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.lstm = nn.LSTM(input_size=64*12, hidden_size=self.n_char_class, 
                            bidirectional=True)
        self.out = nn.Linear(2*self.n_char_class, self.n_char_class)

    def forward(self, x):
        assert x.shape[1] == self.in_channels and x.shape[2] == self.height
        #print(x.shape)
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        # reshaping
        #print(y.shape)
        bs = [y[:,:,:,i].flatten(start_dim=1) for i in range(y.shape[3])] # (T, N, 64* F)
        y = torch.stack(bs)
        #print(y.shape)
        #raise
        # lstm
        y, _ = self.lstm(y)
        y = self.out(y)
        y = torch.log_softmax(y, dim=2)
        return y


class BaseLine1(nn.Module):
    """
    This Model uses a LSTM to work directly on the images columns.
    Jacob recommended this.
    """

    def __init__(self, shape_in=(1, 32, 256), n_char_class=262):
        super().__init__()
        self.shape_in = shape_in
        self.n_char_class = n_char_class
        # in this approach the sequence length is defined by the width of the image
        self.sequence_length = shape_in[-1]
        # a 3 layer bi-LSTM with dopout allowed
        self.lstm = nn.LSTM(input_size=self.shape_in[-2], hidden_size=n_char_class, # output
                            num_layers=3, bidirectional=True, dropout=0.25, # layer specs
                            batch_first=True) # allowing input of shape (N, T, ...)
        # as the LSTM is bidirectional, we get twice the ouput -> Linear layer condenses outputs
        self.out = nn.Linear(2*n_char_class, n_char_class)

    def forward(self, x):
        # x is of shape (N, 1, F, T), with: N = batch size, F = coulm-wise pixels/ features, T = sequence length
        y = x.transpose(3,2) # (N, 1, T, F)
        y = y.squeeze() # (N, T, F)
        # getting the lstms output
        y, _ = self.lstm(y) # (N, T, 2*C), where C = number of char. classes
        y = self.out(y) # (N, T, C)
        # PyTorchs CTCLoss want input of shape (T, N, C) with a log-softmax activation
        y = y.permute(1, 0, 2)
        y = nn.functional.log_softmax(y, dim=2)
        return y


class BaseLine2(nn.Module):
    """
    This model uses a CNN to extract features and a lstm to work on columns of the feature maps.
    """

    def __init__(self, shape_in=(1, 32, 256*4), n_char_class=262):
        super().__init__()
        # CNN layer specs
        conv_layer = lambda c_in, c_out: nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(3,3), padding=1), nn.ReLU(),
                                                       nn.BatchNorm2d(c_out))
        self.shape_in = shape_in
        self.n_char_class = n_char_class
        # The CNN is set up to pool the width, not meddeling with the columns
        self.cnn = nn.Sequential(conv_layer(self.shape_in[0], 32), nn.MaxPool2d(kernel_size=(1,2)),
                                 conv_layer(32, 32), nn.MaxPool2d(kernel_size=(1,2)))
        # as we pool twice, the sequence length is a quater of the images width
        self.sequence_length = int(self.shape_in[-1]/4)
        # a 3 layer bi-LSTM with dopout allowed
        self.lstm = nn.LSTM(input_size=self.shape_in[-2]*32, hidden_size=n_char_class, # output
                            num_layers=3, bidirectional=True, dropout=0.25, # layer specs
                            batch_first=True) # allowing input of shape (N, T, ...)
        # as the LSTM is bidirectional, we get twice the ouput -> Linear layer condenses outputs
        self.out = nn.Linear(2*n_char_class, n_char_class)

    def forward(self, x):
        # x is of shape (N, 1, F, 4*T), with: N = batch size, F = coulm-wise pixels/ features, T = sequence length
        # extracting features/ building feature maps
        y = self.cnn(x) # (N, 32, F, T)
        y = y.transpose(3,2) # (N, 32, T, F)
        bs = [y[:,i] for i in range(y.shape[1])] # (32, N, T, F)
        y = torch.cat(bs, dim=2) # (N, T, 32*F)
        # getting the lstms output
        y, _ = self.lstm(y) # (N, T, 2*C)
        y = self.out(y) # (N, T, C)
        # PyTorchs CTCLoss want input of shape (T, N, C) with a log-softmax activation
        y = y.permute(1, 0, 2)
        y = nn.functional.log_softmax(y, dim=2)
        return y


class BaseLine3(nn.Module):
    """
    This model divides the image into T disjoint sequence-windows.
    A CNN is used to extract features for each window. (same CNN for each window)
    The features are then handed to a LSTM
    """

    def __init__(self, shape_in=(1, 64, 3000), sequence_length=150, n_char_class=262):
        super().__init__()
        # CNN layer specs
        conv_layer = lambda c_in, c_out: nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(3,3), padding=1), nn.ReLU(),
                                                       nn.BatchNorm2d(c_out))
        self.shape_in = shape_in
        self.n_char_class = n_char_class
        self.sequence_length = sequence_length
        # the CNN
        n_features = 128
        assert shape_in[-1] % self.sequence_length == 0, f'width aught to be divisible by {self.sequence_length}'
        self.stride = int(shape_in[-1] / self.sequence_length)
        self.cnn = nn.Sequential(conv_layer(self.shape_in[0], 32), nn.MaxPool2d(kernel_size=(2,2)),
                                 conv_layer(32, 32), nn.MaxPool2d(kernel_size=(2,2)))
        # dense layer to reduce to a fixed output vector length, e.g. length 128
        self.fc = nn.Sequential(nn.Linear(32 * int(self.stride/4 * shape_in[-2]/4), 128), nn.ReLU())
        # as we pool twice, the sequence length is a quater of the images width
        self.sequence_length = sequence_length
        # a 3 layer bi-LSTM with dopout allowed
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_char_class, # output
                            num_layers=3, bidirectional=True, dropout=0.25, # layer specs
                            batch_first=True) # allowing input of shape (N, T, ...)
        # as the LSTM is bidirectional, we get twice the ouput -> Linear layer condenses outputs
        self.out = nn.Linear(2*n_char_class, n_char_class)

    def forward(self, x):
        # extracting features of sliding windows via CNN
        y = []
        for img in x:
            windows = torch.stack([img[:,:,i*self.stride:i*self.stride+self.stride] 
                                   for i in range(self.sequence_length)]) # (T, 1, 32, 32)
            y_i = self.cnn(windows) # (T, 32, H/4, W/4)
            y_i = torch.flatten(y_i, start_dim=1) # (T, 32 * H/4 * W/4)
            y_i = self.fc(y_i) # (T, 128)
            y.append(y_i)
        # LSTM prob. prediction
        y = torch.stack(y) # (N, T, 128)
        y, _ = self.lstm(y) # (N, T, 2*C)
        y = self.out(y) # (N, T, C)
        # converting to CTCLoss input shape
        y = y.permute(1, 0, 2) # (T, N, C)
        y = torch.log_softmax(y, dim=2)
        return y

