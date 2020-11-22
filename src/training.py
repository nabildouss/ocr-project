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
import tqdm
from torchvision.transforms  import Compose, Resize, ToTensor
import numpy as np


class Trainer:

    def __init__(self, model, dset, iterations=int(1e5), s_batch=16, device=torch.device('cpu'), n_workers=4,
                 prog_bar=False):
        self.model = model
        self.dset = dset
        self.iterations = iterations
        self.s_batch = s_batch
        self.device = device
        self.n_workers = n_workers
        self.prog_bar = prog_bar

    def crierion(self):
        return torch.nn.CTCLoss(blank=0).to(self.device)#, zero_infinity=True)
        #return torch.nn.L1Loss()#MSELoss()#CTCLoss(blank=0).to(self.device)#, zero_infinity=True)

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.00005)
        #return torch.optim.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.99), weight_decay=0.00005)
        #return torch.optim.SGD(self.model.parameters(),  lr=0.001, momentum=0.9)#

    def train(self):
        # setting up the training:
        # defined criterion and optimizer
        criterion, optimizer = self.crierion(), self.optimizer()
        # moving the model to the correct (GPU-) device
        self.model.to(self.device)
        if self.prog_bar:
            prog_bar = tqdm.tqdm(total=self.iterations)
        # the training loop
        it_count = 0
        L_IN = [int(self.model.sequence_length) for _ in range(self.s_batch)]
        epoch_count = 1
        last10loss = np.zeros(50)
        while it_count < self.iterations:
            # new data loader required after each epoch
            dloader = DataLoader(self.dset, batch_size=self.s_batch, num_workers=self.n_workers, shuffle=True,
                                 collate_fn=self.dset.batch_transform)
            for batch, targets, l_targets in dloader:
                # forward pass
                batch, embds = batch.to(self.device), embds.to(self.device)
                y = self.model(batch)
                # computing loss and gradients
                loss = criterion(y, targets, L_IN[:len(l_targets)], l_targets)
                #loss = criterion(y, embds)
                loss.backward()
                # optimization step
                optimizer.step()
                # finally increasing the optimization step counter
                it_count += 1
                if it_count >= self.iterations:
                    break
                if self.prog_bar:
                    prog_bar.update(1)
                    last10loss = np.roll(last10loss, 1)
                    last10loss[0] = loss.item()
                    mean_loss = np.mean(last10loss)
                    prog_bar.set_description("epoch %d | mean loss = %f" % (epoch_count, mean_loss / self.s_batch))
            epoch_count += 1
        # moving clearing the GPU memory
        batch.cpu()
        targets.cpu()
        self.model.cpu()


def arg_parser():
    ap = ArgumentParser()
    ap.add_argument('--iterations', default=int(1e5), type=int)
    ap.add_argument('--data_set', default='GT4HistOCR', type=str)
    ap.add_argument('--batch_size', default=20, type=int)
    ap.add_argument('--device', default='cpu', type=str)
    ap.add_argument('--prog_bar', default=True, type=bool)
    ap.add_argument('--out', default=None)
    return ap


def run_training(iterations, data_set, batch_size, device, out, prog_bar, seq_len=256):
    train, _ = ms1.load_data(data_set, n_train=0.75, n_test=0.25,
                             transformation=Compose([Resize([32,3000]), ToTensor()]))
    model = BaseLine(n_char_class=len(train.character_classes)+1, sequence_length=seq_len,
                     shape_in=(1, 32, 32))
    trainer = Trainer(model, train, iterations=iterations, s_batch=batch_size, device=device,
                      prog_bar=prog_bar)
    trainer.train()
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        torch.save(trainer.model.state_dict(), out)


if __name__ == '__main__':
    ap = arg_parser().parse_args()
    run_training(iterations=ap.iterations,  data_set=ap.data_set, batch_size=ap.batch_size, device=ap.device,
                 out=ap.out,  prog_bar=ap.prog_bar)
