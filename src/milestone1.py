from src import data
import matplotlib.pyplot as plt
import os


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


def load_data(data_set='GT4HistOCR', transformation=None, n_train=None, n_test=None, corpora=data.ALL_CORPORA, alphabet=None):
    """
    Generates the data sets

    :param data_set: name of the data set
    :param transformation: transformation to apply to the images
    :param n_train: either number or percentage of training images, None will default to official splits (if they exist)
    :param n_test: either number or percentage of test images, None will default to official splits (if they exist)
    :param corpora: GT4HistOCR specific corpora to be used, defaults to all corpora
    :return: training and test splits
    """
    kwargs_train = {}
    kwargs_test = {}
    if data_set == 'GT4HistOCR':
        # directory of the dataset
        kwargs_train['dset_path'] = os.path.join('..', 'corpus')
        kwargs_test['dset_path'] = os.path.join('..', 'corpus')
        kwargs_train['corpora'] = corpora
        kwargs_test['corpora'] = corpora
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
        # custom transformations
        if transformation is not None:
            kwargs_train['transformation'] = transformation
            kwargs_test['transformation'] = transformation
        # train and test data based on defined parameters
        dset_train = data.GT4HistOCR(**kwargs_train, alphabet=alphabet)
        dset_test = data.GT4HistOCR(**kwargs_test, alphabet=alphabet)
        return dset_train, dset_test
    raise NotImplementedError(f'The dataset {data_set} is unknown.')
