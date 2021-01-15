from torchvision.transforms import Compose, ToTensor, Resize
from argparse import ArgumentParser
import numpy as np
import pickle
import matplotlib as plt
from src.evaluation import *
from src.data import *
import src.milestone1 as ms1
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


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
    f, axs = plt.subplots(n_samples, 3, figsize=[12, 1.5*n_samples])#, gridspec_kw={'wspace': 0, 'hspace': 0})
    f.suptitle(title)
    axs[0][0].set_title('best')
    axs[0][1].set_title('meadian')
    axs[0][2].set_title('worst')

    median = np.median(data)
    s_idcs = np.argsort(data)
    best_idcs = s_idcs[:n_samples]
    worst_idcs = s_idcs[-n_samples:]
    median_idcs = np.abs(np.argsort(data - median))[:n_samples]

    resize = Resize([64, 512])
    for i, row in enumerate(axs):
        img_best = resize(dset[best_idcs[i]][0])
        row[0].axis('off')
        row[0].imshow(img_best[0], cmap='bone', aspect='auto')
        img_meadian = resize(dset[median_idcs[i]][0])
        row[1].axis('off')
        row[1].imshow(img_meadian[0], cmap='bone', aspect='auto')
        img_worst = resize(dset[worst_idcs[i]][0])
        row[2].axis('off')
        row[2].imshow(img_worst[0], cmap='bone', aspect='auto')
    return axs


def images_wer(data, dset, model=None, n_samples=4, title='evalutation'):
    images(data['adj_wer'], dset, model, n_samples, title=f'{title}: WER')


def images_cer(data, dset, model=None, n_samples=4, title='evalutation'):
    images(data['adj_cer'], dset, model, n_samples, title=f'{title}: CER')


def plot_all():
    pass


def parser():
    ap = ArgumentParser()
    ap.add_argument('--corpus_id', default=0, type=int)
    ap.add_argument('--data', default=None, type=str)
    ap.add_argument('--model', default=None, type=str)
    ap.add_argument('--model_type', default='clstm', type=str)
    return ap


if __name__ == '__main__':
    arr1 = np.arange(2000)
    arr2 = np.random.random(2000)
    data = {'adj_wer': arr1, 'adj_cer': arr2}

    ap = parser().parse_args()
    if ap.model_type == 'clstm':
        corpora = [ALL_CORPORA[ap.corpus_id]]
        _, test = ms1.load_data(corpora=corpora, cluster=False, transformation=Compose([ToTensor()]))

        hist(data, title=f'CLSTM: {corpora[0]}')
        plt.show()
        images_wer(data, test, title=f'CLSTM: {corpora[0]}')
        plt.show()
        images_cer(data, test, title=f'CLSTM: {corpora[0]}')
        plt.show()
