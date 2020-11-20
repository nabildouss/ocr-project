from unittest import TestCase
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data import *
from src import sliding_window
import matplotlib.pyplot as plt
from torchvision.transforms  import Compose, Resize, ToTensor
import torch


class TestSlidingWindow(TestCase):

    def setUp(self):
        seq_len = 150
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'),transformation=Compose([Resize([32, 32*seq_len]), ToTensor()]))
        self.sliding_w = sliding_window.SlidingWindow(seq_len=seq_len)

    def test_split_img(self):
        img, _ = self.dset[42]
        sliding_windows = self.sliding_w.sliding_windows(img)
        for w in sliding_windows:
            plt.imshow(w.numpy()[0], cmap='bone')
            plt.show()
