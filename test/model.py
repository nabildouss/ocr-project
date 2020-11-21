from unittest import TestCase
import sys
sys.path.append('..')
from src.model import *
import torch


class TestBaseLine(TestCase):

    def setUp(self):
        self.model = BaseLine(shape_in=(1, 32, 3*32), n_char_class=100, sequence_length=150)

    def testForward(self):
        input = torch.rand(size=(16, 1, 32, 32*150))
        out = self.model(input)

