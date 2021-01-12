import string
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from argparse import ArgumentParser
from src.model import *
from src import ctc_decoder
from src.data import *
from src.baseline import *
import src.milestone1 as ms1
import torch
from torch.utils.data import DataLoader
import tqdm
from torchvision.transforms  import Compose, Resize, ToTensor
import numpy as np


class Trainer:

    def __init__(self, model, dset, iterations=int(1e5), s_batch=16, device=torch.device('cpu'), n_workers=4,
                 prog_bar=False, out='models/mymodel.pth'):
        self.model = model
        self.dset = dset
        self.iterations = iterations
        self.s_batch = s_batch
        self.device = device
        self.n_workers = n_workers
        self.prog_bar = prog_bar
        self.out = out

    def crierion(self):
        return torch.nn.CTCLoss(blank=0, reduction='mean').to(self.device)

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.002)#, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #return torch.optim.SGD(self.model.parameters(),  lr=0.0001, momentum=0.9)

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
        # new data loader initialization required after each epoch 
        dloader = DataLoader(self.dset, batch_size=self.s_batch, num_workers=self.n_workers, shuffle=True,
                             collate_fn=self.dset.batch_transform)
        self.model.train()
        while it_count < self.iterations:
            # epoch loop
            for batch, targets, l_targets in dloader:
                # forward pass
                batch, targets = batch.to(self.device), targets.to(self.device)
                self.model.zero_grad()
                y = self.model(batch)
                # computing loss and gradients
                loss = criterion(y, targets, L_IN[:len(l_targets)], l_targets)
                loss.backward()
                # optimization step
                optimizer.step()
                # fetching a sample and outputting the best path searchs result (should show if overfitting is happening)
                if it_count % 50 == 0:
                    hyp = ctc_decoder.decode(y[:,0,:].detach().cpu().numpy())
                    gt_transcript = targets[:l_targets[0]].detach().cpu().numpy()
                    hyp = torch.tensor(hyp)
                    gt_transcript = torch.tensor(gt_transcript)
                    print('')
                    #print(torch.argmax(y[:,0,:], dim=1)[:50])
                    #print(y[:,0,:][:50])
                    #print(gt_transcript[:50])
                    #print('')
                    print(f'hypthesis: "{self.dset.embedding_to_word(hyp)}"\ngt: "{self.dset.embedding_to_word(gt_transcript)}"')
                # finally increasing the optimization step counter
                it_count += 1
                if it_count >= self.iterations:
                    break
                # saving the model every 10K iterations
                if it_count % 10000 == 0:
                    torch.save(self.model.state_dict(), self.out)
                if self.prog_bar:
                    prog_bar.update(1)
                    last10loss = np.roll(last10loss, 1)
                    last10loss[0] = loss.item()
                    mean_loss = np.mean(last10loss)
                    prog_bar.set_description("epoch %d | mean loss = %f" % (epoch_count, mean_loss))
            # keeping track of the number of epochs
            epoch_count += 1
        # moving clearing the GPU memory
        batch.cpu()
        targets.cpu()
        self.model.cpu()


def arg_parser():
    ap = ArgumentParser()
    ap.add_argument('--iterations', default=int(1e5), type=int)
    ap.add_argument('--data_set', default='GT4HistOCR', type=str)
    ap.add_argument('--data_set_path', default='/home/space/datasets/GT4HistOCR/corpus', type=str)
    ap.add_argument('--corpus_ids', default='12345', type=str) # 0=EarlyModernLatin, 1=Kallimachos, 2=RIDGES_Fraktur, 3=RefCorpus_ENHG_Incunabula, 4=dta19
    ap.add_argument('--batch_size', default=20, type=int)
    ap.add_argument('--device', default='cpu', type=str)
    ap.add_argument('--prog_bar', default=True, type=bool)
    ap.add_argument('--out', default=None)
    ap.add_argument('--corpora_id', default=None)
    return ap


def run_training_1(iterations, data_set, batch_size, device, out, prog_bar, seq_len=256, pixels=32):#seq_len=132
    # sanity checking for the output path
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
    # gathering the training data
    train, _ = ms1.load_data(data_set, n_train=0.75, n_test=0.25,
                             transformation=Compose([Resize([pixels,seq_len]), ToTensor()]),
                             corpora=[Corpus.EarlyModernLatin])
    # setting up the (baseline-) model
    model = BaseLine1(n_char_class=len(train.character_classes)+1, shape_in=(1, pixels, seq_len))
    # initializing training loop
    trainer = Trainer(model, train, iterations=iterations, s_batch=batch_size, device=device,
                      prog_bar=prog_bar, out=out)
    # training
    trainer.train()
    # savong the trained model
    torch.save(trainer.model.state_dict(), out)


def run_training_2(iterations, data_set, batch_size, device, out, prog_bar, seq_len=256, pixels=32):#seq_len=132
    # sanity checking for the output path
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
    # gathering the training data
    train, _ = ms1.load_data(data_set, n_train=0.75, n_test=0.25,
                             transformation=Compose([Resize([pixels,4*seq_len]), ToTensor()]),
                             corpora=[Corpus.EarlyModernLatin])
    # setting up the (baseline-) model
    model = BaseLine2(n_char_class=len(train.character_classes)+1, shape_in=(1, pixels, 4*seq_len))
    # initializing training loop
    trainer = Trainer(model, train, iterations=iterations, s_batch=batch_size, device=device,
                      prog_bar=prog_bar, out=out)
    # training
    trainer.train()
    # savong the trained model
    torch.save(trainer.model.state_dict(), out)


def run_training_3(iterations, data_set, batch_size, device, out, prog_bar, seq_len=256, pixels=32,
                   corpora=ALL_CORPORA):#seq_len=132
    # sanity checking for the output path
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
    # gathering the training data
    train, _ = ms1.load_data(data_set,
                             transformation=Compose([Resize([pixels,pixels*seq_len]), ToTensor()]),
                             corpora=corpora)
    # setting up the (baseline-) model
    model = BaseLine3(n_char_class=len(train.character_classes)+1, shape_in=(1, pixels, pixels*seq_len),
                      sequence_length=seq_len)
    # initializing training loop
    trainer = Trainer(model, train, iterations=iterations, s_batch=batch_size, device=device,
                      prog_bar=prog_bar, out=out)
    # training
    trainer.train()
    # savong the trained model
    torch.save(trainer.model.state_dict(), out)


def run_training_kraken(iterations, data_set, batch_size, device, out, prog_bar, seq_len=256, pixels=48,
                        corpora=ALL_CORPORA):#seq_len=132
    # sanity checking for the output path
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
    # gathering the training data
    train, _ = ms1.load_data(data_set,
                             transformation=Compose([Resize([48,4*seq_len]), ToTensor()]),
                             corpora=corpora)
    #from src.example import ToyData, to_str
    #train = ToyData('toydata')
    #train.batch_transform = train.collate
    #train.embedding_to_word = lambda x: to_str(x, train)
    #model = Kraken(n_char_class=len(train.alphabet))
    # setting up the (baseline-) model
    model = Kraken()
    # initializing training loop
    trainer = Trainer(model, train, iterations=iterations, s_batch=batch_size, device=device,
                      prog_bar=prog_bar, out=out)
    # training
    trainer.train()
    # savong the trained model
    torch.save(trainer.model.state_dict(), out)




if __name__ == '__main__':
    ap = arg_parser().parse_args()
    corpus_ids = [int(c) for c in ap.corpus_ids]
    corpora = [ALL_CORPORA[i] for i in corpus_ids]
    #run_training_1(iterations=ap.iterations,  data_set=ap.data_set, batch_size=ap.batch_size, device=ap.device,
    #               out=ap.out,  prog_bar=ap.prog_bar)
    #run_training_2(iterations=ap.iterations,  data_set=ap.data_set, batch_size=ap.batch_size, device=ap.device,
    #               out=ap.out,  prog_bar=ap.prog_bar)
    run_training_3(iterations=ap.iterations,  data_set=ap.data_set, batch_size=ap.batch_size, device=ap.device,
                   out=ap.out,  prog_bar=ap.prog_bar, corpora=corpora)
    #run_training_kraken(iterations=ap.iterations,  data_set=ap.data_set, batch_size=ap.batch_size, device=ap.device,
    #               out=ap.out,  prog_bar=ap.prog_bar, corpora=corpora)

