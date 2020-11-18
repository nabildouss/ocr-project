from unittest import TestCase
import sys
sys.path.append('..')
from src.model import *
from src.training import *
import src.milestone1 as ms1


class Testtrainer(TestCase):
    """
    This class tests, if the src.training.Trainer's training loop works.
    Only 10 optimization steps are used, so no GPU is needed to execute this test.
    """

    def setUp(self):
        train, _ = ms1.load_data(n_train=0.75, n_test=0.25)
        self.model = BaseLine(n_char_class=len(train.character_classes))
        iterations = 10
        self.trainer = Trainer(self.model, train, iterations=iterations, debug=True)

    def test_train(self):
        self.trainer.train()
