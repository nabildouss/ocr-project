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


WIDE_CHARS = 'ABCDEFGHKLMNOPQRSTUVWXYZ'
WIDE_CHARS += WIDE_CHARS.lower()


class TestPyramidalSlidingWindow(TestCase):

    def setUp(self):
        self.seq_len = 150
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'),
                               transformation=Compose([Resize([32, 32*self.seq_len]), ToTensor()]))
        self.sliding_w = sliding_window.PyramidalSlidingWindow(seq_len=self.seq_len)

    def test_split_img(self):
        # data
        img, transcript = self.dset[np.random.randint(0, len(self.dset))]
        # get the sliding windows
        sliding_windows = self.sliding_w.sliding_windows(img)
        # estimate alignment
        char_weights = []
        for c in transcript:
            if c in WIDE_CHARS:
                char_weights.append(1)
            else:
                char_weights.append(0.5)
        char_weights = np.array(char_weights)
        scale_factor = self.seq_len / np.sum(char_weights)
        char_weights *= scale_factor
        offset = 0
        chars = []
        j = 0
        for i in range(self.seq_len):
            if i < offset + char_weights[j]:
                chars.append(transcript[j])
            else:
                offset += char_weights[j]
                j += 1

            #char_width = int(self.seq_len/len(transcript))
            #chars = np.empty(self.seq_len, dtype=str)
            #for k in range(len(transcript)):
            #    chars[k*char_width:k*char_width+char_width] = transcript[k]
        # display
        i = 0
        plt.imshow(img[0], cmap='bone')
        plt.show()
        for w in sliding_windows[:25]:
            f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
            print(w.shape)
            ax1.imshow(w.numpy()[0, 0], cmap='bone')
            ax2.imshow(w.numpy()[1, 0], cmap='bone')
            ax3.imshow(w.numpy()[2, 0], cmap='bone')
            ax4.text(0, 0, chars[i], fontsize=128)
            plt.show()
            plt.close(f)
            i += 1


class TestSlidingWindow(TestCase):

    def setUp(self):
        self.seq_len = 150
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'),
                               transformation=Compose([Resize([24, 24*self.seq_len]), ToTensor()]))
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
        for w in sliding_windows[:35]:
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
        for idx in np.random.randint(low=0, high=len(self.dset), size=5):
            img, transcript = self.dset[idx]
            sliding_windows = self.sliding_w.sliding_windows(img)
            plt.imshow(img.numpy()[0, :, :500])
            plt.show()
            for i, w in enumerate(sliding_windows[:3]):
                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(w.numpy()[0], cmap='bone')
                ax2.text(0, 0, transcript[i], fontsize=64)
                plt.show()
                plt.close(f)
