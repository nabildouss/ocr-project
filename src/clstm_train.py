from argparse import ArgumentParser
from numpy import array, zeros
import tqdm
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data import *
import src.milestone1 as ms1
import matplotlib.pyplot as plt
import cv2
import clstm


def mktarget(transcript, noutput):
    N = len(transcript)
    target = zeros((2*N+1, noutput),'f')
    assert 0 not in transcript
    target[0, 0] = 1
    for i, c in enumerate(transcript):
        target[2*i+1, c] = 1
        target[2*i+2, 0] = 1
    return target


def run_training(net, dset, n_workers=4, iterations=4e4):
    dloader = DataLoader(dset, batch_size=1, num_workers=n_workers, shuffle=True,
                         collate_fn=dset.batch_transform)
    noutput = len(dset.character_classes)+1
    it_count = 0
    prog_bar = tqdm.tqdm(total=iterations)
    while it_count < iterations:
        for batch, targets, l_targets in dloader:
            # conversion to CLSTM framework inputs
            x_in = batch.cpu().detach().numpy().reshape(*batch.shape[-2:]).T
            scale_factor = 32 / x_in.shape[1]
            x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
            x_in = x_in[:, :, None]
            y_target = mktarget(targets, noutput=noutput)
            #plt.imshow(y_target, cmap='bone')
            #plt.show()
            #plt.imshow(x_in.reshape(-1,32).T, cmap='bone')
            #plt.show()
            #raise
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
            ## actual gradient adjustment
            deltas = aligned - y_pred
            # backward
            net.d_outputs.aset(deltas)
            net.backward()
            # optimization step
            net.update()
            # increasing the iteration counter
            it_count += 1
            prog_bar.update(1)
            if it_count >= iterations:
                break
    return net


def save(net, p_out):
    if not os.path.isdir(os.path.dirname(p_out)):
        os.makedirs(os.path.dirname(p_out))
    clstm.save_net(p_out, net)



def parser():
    ap = ArgumentParser()
    ap.add_argument('--corpus_id', default=1, type=int)
    ap.add_argument('--out', default='CLSTM_models/my_model.clstm', type=str)
    ap.add_argument('--iterations', default=int(4e4), type=int)
    return ap


if __name__ == '__main__':
    # argument parsing
    ap = parser().parse_args()
    corpora = [ALL_CORPORA[ap.corpus_id]]
    train, _ = ms1.load_data(corpora=corpora, cluster=True, transformation=Compose([ToTensor()]))
    #train, _ = ms1.load_data(corpora=corpora, cluster=False, transformation=Compose([ToTensor()]))
    # construct network
    ninput = 32
    noutput = len(train.character_classes)+1
    net = clstm.make_net_init("bidi","ninput=%d:nhidden=100:noutput=%d"%(ninput,noutput))
    net.setLearningRate(1e-4, 0.9)
    # training
    net = run_training(net, train, iterations=ap.iterations)
    # saving the network
    save(net, p_out=ap.out)
