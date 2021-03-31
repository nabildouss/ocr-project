from torchvision.transforms import Compose, ToTensor, Resize
from argparse import ArgumentParser
import numpy as np
import torch

from copy import deepcopy
import pickle
import matplotlib as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.evaluation import *
from src.baseline import *
from src.data import *
import src.milestone1 as ms1
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import captum


def hist(data, bins=100, title='evaluation'):
    wer = data['adj_wer']
    cer = data['adj_cer']
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5])
    f.suptitle(title)
    ax1.set_xlabel('CER')
    ax1.set_ylabel('percentage of test images')
    p_wer = sns.histplot(data=wer, kde=False, ax=ax1, bins=bins, stat='probability')
    ax2.set_xlabel('WER')
    ax2.set_ylabel('percentage of test images')
    p_cer = sns.histplot(data=cer, kde=False, ax=ax2, bins=bins, stat='probability')
    return p_wer, p_cer


def images(data, dset, model=None, n_samples=4, title='evalutation'):
    #if model is None:
    f, axs = plt.subplots(n_samples+1, 3, figsize=[18, 1.5*(n_samples+1)])#, gridspec_kw={'wspace': 0, 'hspace': 0})
    #else:
    #    f, axs = plt.subplots(2*n_samples, 3, figsize=[18, 1.5*n_samples])#, gridspec_kw={'wspace': 0, 'hspace': 0})
    f.suptitle(title)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0][0].text(0.5, 0.5, 'best', horizontalalignment='center', verticalalignment='center', fontsize=24)
    axs[0][0].axis('off')
    axs[0][1].text(0.5, 0.5, 'median', horizontalalignment='center', verticalalignment='center', fontsize=24)
    axs[0][1].axis('off')
    axs[0][2].text(0.5, 0.5, 'worst', horizontalalignment='center', verticalalignment='center', fontsize=24)
    axs[0][2].axis('off')

    median = np.median(data)
    s_idcs = np.argsort(data)
    best_idcs = s_idcs[:n_samples]
    worst_idcs = s_idcs[-n_samples:]
    median_idcs = np.argsort(np.abs(data - median))[:n_samples]

    resize = Resize([64, 720])
    for i, row in enumerate(axs[1:]):
        img_best = resize(dset[best_idcs[i]][0])
        row[0].axis('off')
        row[0].imshow(img_best[0], cmap='bone', aspect='auto')
        img_meadian = resize(dset[median_idcs[i]][0])
        row[1].axis('off')
        row[1].imshow(img_meadian[0], cmap='bone', aspect='auto')
        img_worst = resize(dset[worst_idcs[i]][0])
        row[2].axis('off')
        row[2].imshow(img_worst[0], cmap='bone', aspect='auto')
        if model is not None:
            hyp_best = model[best_idcs[i]]
            hyp_median = model[median_idcs[i]]
            hyp_worst = model[worst_idcs[i]]
            # if isinstance(model, BaseLine3) or isinstance(model, Kraken):
            #     hyp_best = pred_baseline_models(torch.stack([dset[best_idcs[i]][0]]), model, dset)
            #     hyp_median = pred_baseline_models(torch.stack([dset[median_idcs[i]][0]]), model, dset)
            #     hyp_worst = pred_baseline_models(torch.stack([dset[worst_idcs[i]][0]]), model, dset)
            # else:
            #     hyp_best = pred_clstm(torch.stack([dset[best_idcs[i]][0]]), model, dset)
            #     hyp_median = pred_clstm(torch.stack([dset[median_idcs[i]][0]]), model, dset)
            #     hyp_worst = pred_clstm(torch.stack([dset[worst_idcs[i]][0]]), model, dset)
            row[0].set_title(hyp_best, fontsize=12)
            row[1].set_title(hyp_median, fontsize=12)
            row[2].set_title(hyp_worst, fontsize=12)
    return axs


def images_wer(data, dset, model=None, n_samples=1, title='evalutation'):
    images(data['adj_wer'], dset, model, n_samples, title=f'{title} WER')


def images_cer(data, dset, model=None, n_samples=1, title='evalutation'):
    images(data['adj_cer'], dset, model, n_samples, title=f'{title} CER')


def plot_all(data, test, pfx, corpus, dir_out, model=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)
    plt.tight_layout()
    hist(data, title=f'{pfx}: {corpus.value}')
    plt.savefig(os.path.join(dir_out, f'{pfx}: {corpus.value}_hist'))

    images_wer(data, test, title=f'{pfx}: {corpus.value}', model=model)
    plt.savefig(os.path.join(dir_out, f'{pfx}: {corpus.value}_imgsWER'))

    images_cer(data, test, title=f'{pfx}: {corpus.value}', model=model)
    plt.savefig(os.path.join(dir_out, f'{pfx}: {corpus.value}_imgsCER'))


def parser():
    ap = ArgumentParser()
    ap.add_argument('--corpus_id', default=0, type=int)
    ap.add_argument('--data', default=None, type=str)
    ap.add_argument('--model', default=None, type=str)
    ap.add_argument('--model_type', default='clstm', type=str)
    ap.add_argument('--dir_out', default='CLSTM_plots', type=str)
    return ap


# def pred_clstm(batch, net, dset):
#     x_in = batch.cpu().detach().numpy().reshape(*batch.shape[-2:]).T
#     scale_factor = 32 / x_in.shape[1]
#     x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
#     x_in = x_in[:, :, None]
#     net.inputs.aset(x_in)
#     y_pred = net.outputs.array()
#     hyp = ctc_decoder.decode(y_pred.reshape(-1, len(dset.char_classes)+1))
#     hyp = torch.tensor(hyp)
#     hyp = dset.embedding_to_word(hyp)
#     return hyp
#
#
# def pred_baseline_models(batch, net, dset):
#     y_pred = net(batch)
#     hyp = ctc_decoder.decode(y_pred[:, 0].cpu().detach().numpy())
#     hyp = torch.tensor(hyp)
#     hyp = dset.embedding_to_word(hyp)
#     return hyp


if __name__ == '__main__':
    ap = parser().parse_args()
    data = pickle.load(open(ap.data, 'rb'))#{'adj_wer': arr1, 'adj_cer': arr2}
    corpora = [ALL_CORPORA[ap.corpus_id]]
    _, test = ms1.load_data(corpora=corpora, cluster=False, transformation=Compose([ToTensor()]))
    preds = None
    if ap.model is not None:
        preds = pickle.load(open(ap.model, 'rb'))
    if ap.model_type == 'clstm':
        pfx = 'CLSTM'
    elif ap.model_type == 'baseline':
        pfx = 'Sliding Windows'
            #model = BaseLine3()
            ## setting up the (baseline-) model
            #_, test = ms1.load_data(transformation=Compose([Resize([32, 32 * 256]), ToTensor()]),
            #                        corpora=corpora, cluster=False)
            ## setting up the (baseline-) model
            #model = BaseLine3(n_char_class=len(test.character_classes) + 1, shape_in=(1, 32, 32 * 256),
            #                  sequence_length=256)
            ## loading the model
            #state_dict = torch.load(ap.model, map_location=torch.device('cpu'))
            #model.load_state_dict(state_dict=state_dict)
            #model.eval()
    elif ap.model_type == 'kraken':
        pfx = 'Lightweight Model'
        # setting up the (baseline-) model
        #_, test = ms1.load_data(transformation=Compose([Resize([48, 4 * 256]), ToTensor()]),
                                #corpora=corpora, cluster=False)
        #model = Kraken(n_char_class=len(test.character_classes) + 1)
        ## loading the model
        #state_dict = torch.load(ap.model, map_location=torch.device('cpu'))
        #model.load_state_dict(state_dict=state_dict)
        #model.eval()
    else:
        raise ValueError(f'unkown model type {ap.model_type}')
    plot_all(data, test, pfx, corpora[0], ap.dir_out, model=preds)


def confidence_plot(cer, confs, save_path=None, show=False):
    plt.clf()
    dct = {'CER': cer, 'confidence': confs}
    data = pd.DataFrame.from_dict(dct)
    plot = sns.displot(data, x='CER', y='confidence', facet_kws={'xlim':(0,1), 'ylim':(0,1)}, palette='dark', cbar=True,
                       color='black', cbar_kws={'drawedges': False}, discrete=False)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def local_importance(grads, width=32):
    adjusted = np.zeros(grads.shape)
    #for i in range(grads.shape[1] // width):
    #    adjusted[:, i*width: (i+1)*width] = grads[:, i*width: (i+1)*width] / np.sum(grads[:, i*width: (i+1)*width])
    k_size = width
    for i in range(grads.shape[0] // k_size):
        for j in range(grads.shape[1] // k_size):
            adjusted[i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size] = grads[i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size] / np.sum(grads[i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size])
    return adjusted
 

def explanation_plot(input, model, targets, L_IN, l_targets, framework='torch', criterion=None, save_path=None, show=False):
    if framework == 'torch':
        #print(input.shape, targets.shape, L_IN, l_targets)
        if criterion is None:
            criterion = torch.nn.CTCLoss()
        input = torch.autograd.Variable(input, requires_grad=True)
        y = model(input)
        torch.nn.CTCLoss(blank=0, reduction='mean')
        loss = criterion(y, targets, L_IN, l_targets)
        loss.backward()
        WIDTH = 512
        pos_grad = deepcopy(input.grad.data)
        pos_grad *= pos_grad > 0
        
    else:
        from src import clstm_train
        import clstm
        x_in = input.cpu().detach().numpy().reshape(*input.shape[-2:]).T
        scale_factor = 32 / x_in.shape[1]
        x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
        x_in = x_in[:, :, None]
        y_target = clstm_train.mktarget(targets, noutput=262)
        # plt.imshow(y_target, cmap='bone')
        # plt.show()
        # plt.imshow(x_in.reshape(-1,32).T, cmap='bone')
        # plt.show()
        # raise
        # forward pass
        input = x_in
        model.inputs.aset(x_in)
        model.forward()
        y_pred = model.outputs.array()
        # gradients
        ## alignments
        seq = clstm.Sequence()
        seq.aset(y_target.reshape(-1, 262, 1))
        aligned = clstm.Sequence()
        clstm.seq_ctc_align(aligned, model.outputs, seq)
        aligned = aligned.array()
        ## actual gradient adjustment
        deltas = aligned - y_pred
        # backward
        model.d_outputs.aset(deltas)
        model.backward()

        pos_grad = model.d_inputs.array()
        pos_grad *= (pos_grad>0)

    pos_grad_x_inp = input * pos_grad
    if save_path is not None:
        if not isinstance(input, np.ndarray):
            input = input.detach().cpu().numpy()
        if not isinstance(pos_grad_x_inp, np.ndarray):
            pos_grad_x_inp = pos_grad_x_inp.detach().cpu().numpy()
        if framework != 'torch':
            input = np.squeeze(input)
            input = input.T
            print(input.shape)
            pos_grad_x_inp = np.squeeze(pos_grad_x_inp )
            pos_grad_x_inp = pos_grad_x_inp.T
            print(pos_grad_x_inp.shape)
        input_img = np.squeeze(input)
        plt.clf()
        X = np.arange(input_img.shape[1])
        Y = np.mean(input_img, axis=0)
        plt.plot(X, Y)
        plt.xlabel('sequence step')
        plt.ylabel('mean of grad x input')
        plt.savefig(save_path + 'grad_mean.png')
        input /= input_img.max()
        input *= 255
        pos_grad_x_inp_img = np.squeeze(pos_grad_x_inp)
        pos_grad_x_inp_img /= pos_grad_x_inp_img.max()
        pos_grad_x_inp_img *= 255
        input_img = cv2.resize(input_img, (512, 32))
        pos_grad_x_inp_img = cv2.resize(pos_grad_x_inp_img, (512, 32))
        cv2.imwrite(save_path + 'input_img.png', input_img)
        cv2.imwrite(save_path + 'grad_img.png', pos_grad_x_inp_img)
    return pos_grad_x_inp


def len_plot(cer, lengths, save_path, bin_len=20, xlabel='CER'):
    plt.clf()
    cer = np.array(cer)
    lengths = np.array(lengths)
    idcs = np.argsort(cer)
    cer = cer[idcs]
    lengths = lengths[idcs]
    
    l, e = [], []
    X = []
    Y = []
    i = 0
    for k in range(len(cer)):
        if i >= bin_len or k == len(cer)-1:
            X.append(np.mean(e))
            Y.append(np.mean(l))
            e = []
            l = []
            i = 0
        else:
            l.append(lengths[k])
            e.append(cer[k])
            i += 1
    plt.plot(X, Y)
    plt.xlabel(xlabel)
    plt.ylabel('length')
    plt.savefig(save_path)


def corrections_plot(err, save_path):
    plt.clf()
    err = np.array(err)
    idcs = np.argsort(err)
    err = err[idcs][::-1]
    corrections = np.arange(len(err))
    ERR = []
    ticks = []
    under_20 = False
    under_10 = False
    under_5 = False
    for i in range(len(err)):
        new_acc = 1-(np.sum(err[i:])/len(err))
        if new_acc >= 0.8 and not under_20:
            under_20 = True
            ticks.append(i)
        if new_acc >= 0.9 and not under_10:
            under_10 = True
            ticks.append(i)
        if new_acc >= 0.95 and not under_5:
            under_5 = True
            ticks.append(i)
        ERR.append(new_acc)
    plt.plot(corrections, ERR)
    for t in ticks:
        plt.vlines(t, np.amin(ERR), np.amax(ERR), 'r', 'dotted')
    #plt.xticks(list(plt.xticks()[0]) + ticks)
    plt.xlabel(f'# corrections, breaks at {ticks}')
    plt.ylabel('accuracy')
    plt.savefig(save_path)


def threshold_plot(err, confs, save_path, ylabel='error rate'):
    plt.clf()
    threshs = np.arange(start=0.5, stop=1, step=0.1)
    err = np.array(err)
    confs = np.array(confs)
    
    e_rates = []
    for t in threshs:
        e_rates.append(np.sum(err[confs>t]) / len(err))
    plt.plot(threshs, e_rates)
    plt.xlabel('threshold')
    plt.ylabel(ylabel)
    plt.savefig(save_path)
