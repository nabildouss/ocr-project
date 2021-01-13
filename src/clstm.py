from argparse import ArgumentParser
from numpy import array, zeros
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data import *
import src.milestone1 as ms1
import matplotlib.pyplot as plt
# import clstm


def mktarget(transcript, noutput):
    N = len(transcript)
    target = zeros((2*N+1, noutput),'f')
    assert 0 not in transcript
    target[0, 0] = 1
    for i, c in enumerate(transcript):
        target[2*i+1, c] = 1
        target[2*i+2, 0] = 1
    return target


def run_training(net, dset, n_workers=4):
    dloader = DataLoader(dset, batch_size=1, num_workers=n_workers, shuffle=True,
                         collate_fn=dset.batch_transform)
    noutput = len(dset.character_classes)+1
    for batch, targets, l_targets in dloader:
        # conversion to CLSTM framework inputs
        x_in = batch.cpu().detach().numpy().reshape(batch.shape[-2:]).T
        y_target = mktarget(targets, noutput=noutput)
        plt.imshow(y_target, cmap='bone')
        plt.show()
        plt.imshow(x_in.T, cmap='bone')
        plt.show()
        raise
        # forward pass
        net.inputs.aset(x_in)
        net.forward()
        y_pred = net.outputs.array()
        # gradients
        ## alignments
        seq = clstm.Sequence()
        seq.aset(y_target.reshape(-1, noutput, 1))
        aligned = clstm.Sequence()
        clstm.seq_ctc_align(aligned, net.outputs, seq)
        aligned = aligned.array()
        imshow(aligned.reshape(-1, noutput).T, interpolation='none')
        ## actual gradient adjustment
        deltas = aligned - y_pred
        # backward
        net.d_outputs.aset(deltas)
        net.backward()
        # optimization step
        net.update()
    return net


def save(net, p_out):
    if not os.path.isdir(os.path.dirname(p_out)):
        os.makedirs(os.path.dirname(p_out))
    # clstm.save_net(p_out, net)



def parser():
    ap = ArgumentParser()
    ap.add_argument('--corpus_id', default=1, type=int)
    ap.add_argument('--out', default='CLSTM_models/my_model', type=str)
    return ap


if __name__ == '__main__':
    # argument parsing
    ap = parser().parse_args()
    corpora = [ALL_CORPORA[ap.corpus_id]]
    train, _ = ms1.load_data(corpora=corpora, cluster=False)
    # construct network
    net = None
    # training
    net = run_training(net, train)
    # saving the network
    save(net)
