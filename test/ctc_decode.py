import numpy as np
from unittest import TestCase
from src import ctc_decoder


class TestCTCT_decode_functions(TestCase):

    def setUp(self):
        self.chars = 'abc'
        self.probs = np.array([[0.1, 0.5,  0.1, 0.3],
                               [0.7, 0.1, 0.1, 0.1],
                               [0.1, 0.7, 0.1,  0.1],
                               [0.2, 0.1, 0.6, 0.1]])
        self.word = 'aab'

    def test_decode(self):
        decoded, _ =  ctc_decoder.decode(self.probs)
        str_decoded = ''.join([self.chars[i-1] for i in decoded])
        print(str_decoded)
        self.assertEqual(str_decoded,  self.word)

