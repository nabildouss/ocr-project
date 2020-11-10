"""
This module provides tests w.r.t. the src.data module

Unittests can be executed via your IDE of choice or directly from terminal, e.g. python3 -m unittest data.py
"""
from unittest import TestCase
import sys
sys.path.append('..')
from src.data import *
import numpy as np
from torch.utils.data import DataLoader




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
        d_loader = DataLoader(dataset=self.dset, num_workers=4, batch_size=16)
        # loading all images in an epoch
        for imgs_batch, transcripts_batch in d_loader:
            pass

    def test_display_img(self):
        # considering 10 random but unique images and their transcriptions
        idcs = np.random.permutation(np.arange(len(self.dset))[:10])
        for idx in idcs:
            self.dset.display_img(idx, show=True)
