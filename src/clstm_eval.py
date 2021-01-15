import torch
from argparse import ArgumentParser
from numpy import array, zeros
import tqdm
from torch.utils.data import DataLoader
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data import *
from src.evaluation import *
from src import ctc_decoder
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


def run_eval(net, dset, n_workers=4):
    dloader = DataLoader(dset, batch_size=1, num_workers=n_workers, shuffle=True,
                         collate_fn=dset.batch_transform)
    noutput = len(dset.character_classes)+1
    l_wer = []
    l_cer = []
    l_adj_wer = []
    l_adj_cer = []
    it_count = 0
    prog_bar = tqdm.tqdm(total=len(dset))
    for batch, targets, l_targets in dloader:
        # conversion to CLSTM framework inputs
        x_in = batch.cpu().detach().numpy().reshape(*batch.shape[-2:]).T
        scale_factor = 32 / x_in.shape[1]
        x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
        x_in = x_in[:, :, None]
        y_target = mktarget(targets, noutput=noutput)
        # forward pass
        net.inputs.aset(x_in)
        net.forward()
        y_pred = net.outputs.array()
        # evaluate prediction
        hyp = ctc_decoder.decode(y_pred.reshape(-1, noutput))
        gt_transcript = targets[:l_targets[0]].detach().cpu().numpy()
        hyp = torch.tensor(hyp)
        gt_transcript = torch.tensor(gt_transcript)
        hyp = dset.embedding_to_word(hyp)
        ref = dset.embedding_to_word(gt_transcript)
        l_wer.append(wer(ref, hyp))
        l_cer.append(cer(ref, hyp))
        l_adj_wer.append(adjusted_cer(ref, hyp))
        l_adj_cer.append(adjusted_cer(ref, hyp))
        # increasing the iteration counter
        it_count += 1
        prog_bar.update(1)
    data = {'adj_wer': l_adj_wer, 'adj_cer': l_adj_cer}
    return map(np.mean, [l_wer, l_adj_wer, l_cer, l_adj_cer]), data


def load(p_out):
    return clstm.load_net(p_out)



def parser():
    ap = ArgumentParser()
    ap.add_argument('--clstm_path', default='', type=str)
    ap.add_argument('--corpus_id', default=1, type=int)
    ap.add_argument('--out', default='CLSTM_results/my_model.json', type=str)
    return ap


if __name__ == '__main__':
    # argument parsing
    ap = parser().parse_args()
    corpora = [ALL_CORPORA[ap.corpus_id]]
    _, test = ms1.load_data(corpora=corpora, cluster=True, transformation=Compose([ToTensor()]))
    #_, test = ms1.load_data(corpora=corpora, cluster=False, transformation=Compose([ToTensor()]))
    # construct network
    ninput = 32
    noutput = len(test.character_classes)+1
    net = load(ap.clstm_path)
    # evaluation
    (wer, adj_wer, cer, adj_cer), data = run_eval(net, test)
    summary = {'wer': wer, 'adj_wer': adj_wer, 'cer': cer, 'adj_cer': cer}
    # storing the dictionary as a JSON file
    if not  os.path.isdir(os.path.dirname(ap.out)):
        os.makedirs(os.path.dirname(ap.out))
    with open(ap.out, 'w') as f_out:
        json.dump(summary, f_out)
    with open(pth_out + '_data.pkl', 'wb') as f_data:
        pickle.dump(data, f_data)
    # finally printing the results
    print(summary)
