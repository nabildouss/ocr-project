from unittest import TestCase
import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath(__file__))
import src.milestone1 as ms1
import numpy as np
from torch.utils.data import DataLoader
from src.data import *


class TestMS1(TestCase):
    """
    This class features tests for the Milestone 1 methods
    """

    def test_show(self):
        imgs = np.random.randint(0, 255, size=(6, 64, 512))
        ms1.show(imgs)

    def test_load_data(self):
        # percentages
        dset, _ = ms1.load_data('GT4HistOCR')
        N = len(dset)
        n_train = int(N*0.75)
        n_test = int(N*0.25)
        train, test = ms1.load_data(n_train=0.75, n_test=0.25)
        self.assertEqual(len(train), n_train)
        self.assertEqual(len(test), n_test)
        # discrete numbers of samples
        train, test = ms1.load_data(n_train=750, n_test=250)
        self.assertEqual(len(train), 750)
        self.assertEqual(len(test), 250)

    def test_dloader(self):
        train, test = ms1.load_data(n_train=0.75, n_test=0.25)
        d_loader = DataLoader(dataset=test, num_workers=4, batch_size=16, collate_fn=test.batch_transform)
        for batch, targets, l_targets in d_loader:
            pass

    def test_default_splits(self):
        for c in ALL_CORPORA:
            train, test = ms1.load_data('GT4HistOCR', corpora=[c])
            print(len(train), len(test), len(train)+len(test))

