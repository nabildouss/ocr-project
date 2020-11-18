"""
This module provides tests w.r.t. the src.data module

Unittests can be executed via your IDE of choice or directly from terminal, e.g. python3 -m unittest data.py
"""
from unittest import TestCase
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data import *
import numpy as np
from torch.utils.data import DataLoader
import string




class TestGT4HistOCR(TestCase):
    """
    The tests only concern the length of the image and whether all images of an epoch can be loaded.
    Evidence of the Images matching their transcriptions can be obtained by test_display_img.
    """

    def setUp(self):
        self.dset = GT4HistOCR(os.path.join('..', 'corpus'))

    def test_len(self):
        # my version of the dataset contains a couple images more, maybe the 313173 of the README is outdated...
        # self.assertEqual(len(self.dset), 313173)
        self.assertEqual(len(self.dset), 313209)

    def test_getitem(self):
        # NOTE: as all ~313k images will be read, this method can take 5-6 minutes to complete
        # allowing for 4 workers and batches of size 16 to speed up iterations
        d_loader = DataLoader(dataset=self.dset, num_workers=4, batch_size=16, collate_fn=self.dset.batch_transform)
        # loading all images in an epoch
        for imgs_batch, targets, l_targets in d_loader:
            pass

    def test_display_img(self):
        # considering 10 random but unique images and their transcriptions
        idcs = np.random.permutation(np.arange(len(self.dset))[:10])
        for idx in idcs:
            self.dset.display_img(idx, show=True)

    def test_int_word_mapping(self):
        alphabet = self.dset.character_classes
        for _ in range(10000):
            # considering strings of different lengths
            L = np.random.randint(0, 20)
            # extracting random indices of characters
            idcs = np.random.randint(0, len(alphabet), size=L)
            # generating a random word/ string
            word = ''.join(alphabet[idcs])
            # checking if encoding and decoding yield same results
            self.assertEqual(word, self.dset.embedding_to_word(self.dset.word_to_embedding(word)))
