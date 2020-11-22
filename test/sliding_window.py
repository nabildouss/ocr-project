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
import numpy as np


class TestSlidingWindow(TestCase):

    def setUp(self):
        self.seq_len = 350
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'),
                               transformation=Compose([Resize([32, 32*self.seq_len]), ToTensor()]))
        self.sliding_w = sliding_window.SlidingWindow(seq_len=self.seq_len)

    def test_split_img(self):
        img, transcript = self.dset[np.random.randint(0, len(self.dset))]
        sliding_windows = self.sliding_w.sliding_windows(img)
        char_width = int(self.seq_len/len(transcript))
        chars = np.empty(self.seq_len, dtype=str)
        for k in range(len(transcript)):
            print(transcript[k])
            chars[k*char_width:k*char_width+char_width] = transcript[k]
        i = 0
        for w in sliding_windows[:25]:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(w.numpy()[0], cmap='bone')
            ax2.text(0, 0, chars[i], fontsize=128)
            plt.show()
            plt.close(f)
            i += 1


class TestAutomaticWindows(TestCase):

    def setUp(self):
        self.seq_len = 350
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'),
                               transformation=Compose([ToTensor()]))
        self.sliding_w = sliding_window.AutomaticWindow()

    def test_split_img(self):
        img, transcript = self.dset[42]
        print(transcript)
        sliding_windows = self.sliding_w.sliding_windows(img)
        plt.imshow(img.numpy()[0, :, :500])
        plt.show()
        for i, w in enumerate(sliding_windows[:10]):
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(w.numpy()[0], cmap='bone')
            ax2.text(0, 0, transcript[i], fontsize=64)
            plt.show()
            plt.close(f)
