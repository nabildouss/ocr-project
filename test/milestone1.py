from unittest import TestCase
import sys
sys.path.append('..')
import src.milestone1 as ms1
import numpy as np


class TestMS1(TestCase):

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
