from unittest import TestCase
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import evaluation


class TestEvaluationFunctions(TestCase):

    def setUp(self):
        self.gt = 'hello world how are you'
        self.hypothesis1 = 'oh my world how are you doing'
        self.hypothesis2 = 'hello earth how are you'
        self.hypothesis3 = 'this this this this this this this this this this this this this this this'

    def test_wer(self):
        print(f'WER: {self.hypothesis1}')
        print(evaluation.word_errors(self.gt, self.hypothesis1))
        print(evaluation.wer(self.gt, self.hypothesis1))
        print(evaluation.adjusted_wer(self.gt, self.hypothesis1))
        print(f'CER: {self.hypothesis1}')
        print(evaluation.char_errors(self.gt, self.hypothesis1))
        print(evaluation.cer(self.gt, self.hypothesis1))
        print(evaluation.adjusted_cer(self.gt, self.hypothesis1))

        print(f'WER: {self.hypothesis2}')
        print(evaluation.word_errors(self.gt, self.hypothesis2))
        print(evaluation.wer(self.gt, self.hypothesis2))
        print(evaluation.adjusted_wer(self.gt, self.hypothesis2))
        print(f'CER: {self.hypothesis2}')
        print(evaluation.char_errors(self.gt, self.hypothesis2))
        print(evaluation.cer(self.gt, self.hypothesis2))
        print(evaluation.adjusted_cer(self.gt, self.hypothesis2))

        print(f'WER: {self.hypothesis3}')
        print(evaluation.word_errors(self.gt, self.hypothesis3))
        print(evaluation.wer(self.gt, self.hypothesis3))
        print(evaluation.adjusted_wer(self.gt, self.hypothesis3))
        print(f'CER: {self.hypothesis3}')
        print(evaluation.char_errors(self.gt, self.hypothesis3))
        print(evaluation.cer(self.gt, self.hypothesis3))
        print(evaluation.adjusted_cer(self.gt, self.hypothesis3))
