from unittest import TestCase
import sys
sys.path.append('..')
from src.model import *
import torch


class TestBaseLine(TestCase):

    def setUp(self):
        self.model = BaseLine(shape_in=(1, 64, 512), n_pool_columns=100, n_char_class=100, sequence_length=100)

    def testForward(self):
        input = torch.rand(size=(16, 1, 64, 512))
        out = self.model(input)
