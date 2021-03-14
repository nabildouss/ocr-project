import numpy as np
from unittest import TestCase
from src import visualize


class TestVisualize(TestCase):
	
	def setUp(self):
		self.conf = np.random.normal(0.5, 0.2, 5000)
		self.errs = np.random.normal(0.75, 0.1, 5000)

	def test_confidence_plot(self):
		visualize.confidence_plot(self.errs, self.conf)
