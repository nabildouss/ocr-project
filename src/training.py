import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from argparse import ArgumentParser
from src.model import *
from src.data import *
import src.milestone1 as ms1
import torch
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self, model, dset, iterations=int(1e5), s_batch=16, device=torch.device('cpu'), n_workers=4,
                 debug=False):
        self.debug = debug
        self.model = model
        self.dset = dset
        self.iterations = iterations
        self.s_batch = s_batch
        self.device = device
        self.n_workers = n_workers

    def crierion(self):
        return torch.nn.CTCLoss()

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.99), weight_decay=0.00005)

    def train(self):
        # setting up the training:
        # defined criterion and optimizer
        criterion, optimizer = self.crierion(), self.optimizer()
        # moving the model to the correct (GPU-) device
        self.model.to(self.device)
        # the training loop
        it_count = 0
        while it_count < self.iterations:
            # new data loader required after each epoch
            dloader = DataLoader(self.dset, batch_size=self.s_batch, num_workers=self.n_workers, shuffle=True,
                                 collate_fn=self.dset.batch_transform)
            for batch, targets, l_targets in dloader:
                # moving the data to the (GPU-) device
                batch, targets = batch.to(self.device), targets.to(self.device)
                # forward pass
                y = self.model(batch)
                # computing loss and gradients
                loss = criterion(y, targets, [int(self.model.sequence_length) for _ in range(batch.shape[0])], l_targets)
                loss.backward()
                # optimization step
                optimizer.step()
                # finally increasing the optimization step counter
                it_count += 1
                if it_count >= self.iterations:
                    break
                if self.debug and it_count % 1000 == 0:
                    print(f'Iteration {it_count}:\t{loss}')
        # moving clearing the GPU memory
        batch.cpu()
        targets.cpu()
        self.model.cpu()


def arg_parser():
    ap = ArgumentParser()
    ap.add_argument('--iterations', default=int(1e5), type=int)
    ap.add_argument('--data_set', default='GT4HistOCR', type=str)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--device', default='cpu', type=str)
    ap.add_argument('--out', default=None)
    return ap


if __name__ == '__main__':
    ap = arg_parser().parse_args()
    train, _ = ms1.load_data(ap.data_set, n_train=0.75, n_test=0.25)
    model = BaseLine(n_char_class=len(train.character_classes))
    trainer = Trainer(model, train, iterations=ap.iterations, s_batch=ap.batch_size, device=ap.device)
    trainer.train()
    if ap.out is not None:
        if not os.path.isdir(os.path.dirname(ap.out)):
            os.makedirs(os.path.dirname(ap.out))
        torch.save(trainer.model.state_dict(), ap.out)

