import numpy as np
from unittest import TestCase
from src import visualize, baseline, data
import src.milestone1 as ms1
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import torch
import os


class TestVisualize(TestCase):
    
    def setUp(self):
        self.conf = np.random.normal(0.5, 0.2, 5000)
        self.errs = np.random.normal(0.75, 0.1, 5000)
        pixels = 32
        seq_len = 256
        pth_model = 'test_models/models_100K/sw_0'
        _, test = ms1.load_data(transformation=Compose([Resize([pixels, pixels * seq_len]), ToTensor()]),
                                corpora=[data.ALL_CORPORA[0]], cluster=False)
        dloader = DataLoader(test, batch_size=1, num_workers=4, shuffle=True,
                             collate_fn=test.batch_transform)
        self.model = baseline.BaseLine3(n_char_class=len(test.character_classes) + 1, shape_in=(1, pixels, pixels * seq_len),
                                        sequence_length=seq_len)
        state_dict = torch.load(pth_model, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict=state_dict)
        #self.model.eval()
        for batch, targets, l_targets in dloader:
            self.targets = targets
            self.l_targets = l_targets
            self.input = batch
            break
        self.L_IN = [self.model.sequence_length]
        self.out_file_conf = 'VISUALIZE_TEST.png'
        self.out_file_expl = 'EXPLAIN_TEST.png'

    def tearDown(self):
        if os.path.isfile(self.out_file_conf):
            os.remove(self.out_file_conf)
        if os.path.isfile(self.out_file_expl):
            os.remove(self.out_file_expl)

    def test_confidence_plot(self):
        self.assertTrue(not os.path.isfile(self.out_file_conf))
        visualize.confidence_plot(self.errs, self.conf, save_path=self.out_file_conf, show=True)
        self.assertTrue(os.path.isfile(self.out_file_conf))
    
    def test_explanation_plot(self):
        self.assertTrue(not os.path.isfile(self.out_file_expl))
        visualize.explanation_plot(self.input, self.model, self.targets, self.L_IN, self.l_targets,
                                   save_path=self.out_file_expl, show=True)
        self.assertTrue(os.path.isfile(self.out_file_expl))
