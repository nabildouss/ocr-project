from src import data
import matplotlib.pyplot as plt
import os
import numpy as np
import sys


def show(x, outfile=None):
    """
    Displays images of lines in a table of two columns

    :param x: images of lines
    :param outfile: file to save the figure to, if None the plt.show() will be called
    """
    n_rows = len(x) // 2 + 2 * (len(x) % 2)
    f, axs = plt.subplots(n_rows, 2)
    idx = 0
    for row in axs:
        for ax in row:
            ax.imshow(x[idx])
            idx += 1
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()


def load_data(data_set='GT4HistOCR', transformation=None, n_train=None, n_test=None, corpora=data.ALL_CORPORA
              , alphabet=None, cluster=True):
    """
    Generates the data sets

    :param data_set: name of the data set
    :param transformation: transformation to apply to the images
    :param n_train: either number or percentage of training images, None will default to official splits (if they exist)
    :param n_test: either number or percentage of test images, None will default to official splits (if they exist)
    :param corpora: GT4HistOCR specific corpora to be used, defaults to all corpora
    :param cluster: either the path to dataset in cluster, None will default to local path
    :return: training and test splits
    """
    cluster_path = '/home/space/datasets/GT4HistOCR/corpus'
    kwargs_train = {}
    kwargs_test = {}
    if data_set == 'GT4HistOCR':
        # directory of the dataset
        kwargs_train['dset_path'] = os.path.join('..', 'corpus') if not cluster else cluster_path
        kwargs_test['dset_path'] = os.path.join('..', 'corpus') if not cluster else cluster_path
        kwargs_train['corpora'] = corpora
        kwargs_test['corpora'] = corpora
        # custom transformations
        if transformation is not None:
            kwargs_train['transformation'] = transformation
            kwargs_test['transformation'] = transformation
        # train/ test splits
        if n_train is not None and n_test is not None:
            # n_train and n_test are fixed numbers
            if isinstance(n_train, int) and isinstance(n_test, int):
                f_split_train = lambda x: x[:n_train]
                f_split_test = lambda x: x[n_train:n_train+n_test]
            # n_train and n_test are percentages
            elif isinstance(n_train, float) and isinstance(n_test, float):
                f_split_train = lambda x: x[:int(len(x)*n_train)]
                f_split_test = lambda x: x[int(len(x)*n_train):int(len(x)*n_train)+int(len(x)*n_test)]
            kwargs_train['f_split'] = f_split_train
            kwargs_test['f_split'] = f_split_test
            # train and test data based on defined parameters
            dset_train = data.GT4HistOCR(**kwargs_train, alphabet=alphabet)
            dset_test = data.GT4HistOCR(**kwargs_test, alphabet=alphabet)
        # predefined splits for corpora
        else:
            dset_test, dset_train = data.GT4HistOCR(**kwargs_test, alphabet=alphabet), data.GT4HistOCR(**kwargs_train,
                                                                                                       alphabet=alphabet)
            # discarding the paths for both splits / staring with empty data set
            dset_test.img_paths = np.array([])
            dset_test.gt_paths = np.array([])
            dset_train.img_paths = np.array([])
            dset_train.gt_paths = np.array([])
            # collecting individual corpus splits and concatenating them
            for c in corpora:
                # loading the desired split idcs
                idcs_train = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{c.value}_train.npy'))
                idcs_test = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{c.value}_test.npy'))
                f_split_train = lambda x: x[idcs_train]
                f_split_test = lambda x: x[idcs_test]
                kwargs_train['f_split'] = f_split_train
                kwargs_test['f_split'] = f_split_test
                # split is only ever applicable for the specific corpus only
                kwargs_train['corpora'] = [c]
                kwargs_test['corpora'] = [c]
                # applying the splits on the corpus
                c_test, c_train = data.GT4HistOCR(**kwargs_test, alphabet=alphabet), data.GT4HistOCR(**kwargs_train,
                                                                                                     alphabet=alphabet)
                # concatenating / adding the predefined split to the dataset
                dset_test.img_paths = np.concatenate([c_test.img_paths, dset_test.img_paths])
                dset_test.gt_paths = np.concatenate([c_test.gt_paths, dset_test.gt_paths])
                dset_train.img_paths = np.concatenate([c_train.img_paths, dset_train.img_paths])
                dset_train.gt_paths = np.concatenate([c_train.gt_paths, dset_train.gt_paths])
        return dset_train, dset_test
    raise NotImplementedError(f'The dataset {data_set} is unknown.')
