"""
This module provides classes and methods to load data sets, such as the GT4HistOCR data set.
All classes base torch.utils.data.Dataset, this way the torch.utils.data.DataLoader can be used.
Using the torch DataLoader is important, as we have a lazy loading approach and the DataLoader enables concurrent
workers in python that gather data for batches.
"""
from torchvision.transforms import ToTensor, Resize, Grayscale, Compose
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import Dataset
import torch
from PIL import Image,  ImageFile
from enum import Enum, unique
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt


# allowing for truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


# The GT4HistOCR consists of different corpora, this enum serves for the sake separability
@unique
class Corpus(Enum):
    EarlyModernLatin = 'EarlyModernLatin'
    Kallimachos = 'Kallimachos'
    RIDGES_Fraktur = 'RIDGES-Fraktur'
    RefCorpus_ENHG_Incunabula = 'dta19'
    dta19 = 'RefCorpus-ENHG-Incunabula'


# Most likely we will use all corpora, hence this global constant
ALL_CORPORA = tuple(Corpus)


def search_files(dir_start, suffixes=['.png']):
    """
    Recursively search for all files ending with the defined suffix.
    This method should be used to find images/ gt annotations in a dataset.

    :param dir_start: directory to search in
    :param suffixes: suffixes to search for
    :return: a list of full paths from f to files ending with the suffix
    """
    fs = defaultdict(list)
    # lists all dirs and files in root (note: os.walk is faster than recursion and os.listdir)
    for root,  dirs, files in os.walk(dir_start):
        # sorting file names ensures, that files with same ids in their name have same indices later on
        for f in sorted(files):
            # check if a file ends with a desired suffix
            for k in suffixes:
                if f.endswith(k):
                    fs[k].append(os.path.join(root, f))
    return [fs[k] for k in suffixes]


class OCRDataSet(Dataset):
    """
    This class shall be used as a base class for datasets dealing with OCR.
    """

    def __init__(self, img_paths, gt_paths, f_split=None, transformation=ToTensor(), alphabet=None, invert=True):
        """
        :param img_paths: paths of all images
        :param gt_paths: paths to respective line transcriptions
        :param f_split: a function that filters out data from the img_paths and gt_paths, if None the data set stays whole
        :param img_shape: desired shape of images, if None no resizing takes place
        """
        super().__init__()
        self.invert = invert
        self.img_paths = np.array(img_paths)
        self.gt_paths = np.array(gt_paths)
        # sorting to make splits deterministic
        sorted_idcs = np.argsort(self.img_paths)
        self.img_paths = self.img_paths[sorted_idcs]
        self.gt_paths = self.gt_paths[sorted_idcs]
        # sanity of paths
        assert len(self.gt_paths) == len(self.img_paths), "Number of line image and and annotiation files have to be equal."
        # defining the alphabet of the dataset
        if alphabet is None:
            self.character_classes = self.__character_classes()
        else:
            self.character_classes = alphabet
        self.character_int_map = {c: i+1 for i, c in enumerate(self.character_classes)}
        self.int_chatacter_map = {i+1: c for i, c in enumerate(self.character_classes)}
        # defining the transformations
        self.trans = transformation
        # storing the function, that creates the train/ test splits and applying it
        self.__apply_split(f_split)

    def __getitem__(self, idx):
        """
        :param idx: index of image NOTE: this is NOT the images ID, but simply it's files index in self.img_paths
        :return: Image tensor and embedding of the indexed line
        """
        return self.__load_img(self.img_paths[idx]), self.line(self.gt_paths[idx])

    def __len__(self):
        return len(self.gt_paths)

    def __load_img(self, img_path):
        """
        This method should be the fastest way to load an image and convert it to a tensor.

        :param img_path: Path to the image
        :return: Image tensor (black and white/ i.e. one channel)
        """
        img = Image.open(img_path)
        tensor = self.trans(img)
        # grayscale conversion in case of RGB (3 channels) or RGB + opacity (4 channels) image representations
        if tensor.shape[0] > 1:
            tensor = self.grayscale(tensor[:3, :, :])
        # normalizae lighting
        tensor -=  tensor.min()
        tensor /=  tensor.max()
        if self.invert:
            tensor = 1-tensor
        return tensor

    def grayscale(self, x):
        return rgb_to_grayscale(x, num_output_channels=1)

    def line(self, gt_path):
        """
        :param gt_path: path to the txt file containing the lines transcript
        :return: the lines transcript
        """
        with open(gt_path, 'r') as f_gt:
            line = ''.join(f_gt.readlines())
        line = line.rstrip()
        return line

    def __apply_split(self, f_split):
        """
        Applying the split function, filtering out paths
        """
        if f_split is not None:
            self.img_paths = f_split(self.img_paths)
            self.gt_paths = f_split(self.gt_paths)

    def __character_classes(self):
        """
        extracts the set of all characters in the dara set

        :return: array of all character-classes
        """
        c_set = set()
        for pth in self.gt_paths:
            line = self.line(pth)
            c_set = c_set.union(set(line))
        c_list = sorted(c_set)
        return np.array(c_list)

    def batch_transform(self, data):
        """
        This method is of special importance.
        The CTCLoss requires input of different length for the targets, this is problematic, as a tensors elements
        have to be of equal dimensionality. Also we can not work on characters, but integers.
        To solve this problem, we map all characters to integers (see: self.word_to_embedding) and pad these vectors.
        To allow CTCLoss to work on unpadded, we pass a list of the original lengths.

        :param data: image - transcription pairs
        :return: batch of images, padded targets and their lengths
        """
        batch, targets, l_targets = [], [], []
        for img, transcript in data:
            # images do not need to be changed
            batch.append(img)
            # mapping characters to integers
            targets.append(self.word_to_embedding(transcript))
            # keeping track of ooriginal lengths
            l_targets.append(len(transcript))
        # Tensor conversion for batch and targets, the lengths can stay as a list
        #return torch.stack(batch), torch.nn.utils.rnn.pad_sequence(targets, batch_first=True), l_targets
        return torch.stack(batch), torch.cat(targets), l_targets

    def word_to_embedding(self, word):
        """
        Takes a word and encodes it into a CTC representation

        :param word: string of word
        :return: embedding / mapping to ints
        """
        return torch.from_numpy(np.array([self.character_int_map[c] for c in word], dtype=np.int_))

    def embedding_to_word(self, emb, blank=0):
        """
        Takes an embedding and decodes it into a string representation

        :param emb: sequence of integer values of a word
        :return: string / word representation
        """
        to_chars = [self.int_chatacter_map[int(i.detach().numpy())] for i in emb if i.detach().numpy() != blank]
        if to_chars == []:
            return ''
        return ''.join(to_chars)

    def display_img(self, idx, show=False, ax=None):
        """
        Displaying the image at a given index

        :param idx: index of image file
        :param show: boolean indicating whether to display the image right away
        :return: matplotlib object of the image-plot
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        img, transcript = self[idx]
        ax_im = ax.imshow(img[0], cmap='bone')
        ax.set_title(transcript)
        if show:
            plt.show()
        return ax_im

    def max_mean_seq_length(self):
        max_len =  0
        mean_len =  0
        for pth in self.gt_paths:
            L = len(self.line(pth))
            max_len = max(L,  max_len)
            mean_len += L
        mean_len /= len(self.gt_paths)
        return max_len, mean_len


class GT4HistOCR(OCRDataSet):
    """
    This class adapts the GT4HistOCR data set structure to a torch Dataset.
    I have not found any official train/ test/ val splits, hence no splitting function is defined.
    For now we consider that the dataset is used for pretraining untill we discover official splits.
    """

    def __init__(self, dset_path, corpora=ALL_CORPORA, f_split=None,
                 transformation=Compose([Resize([64, 512]), ToTensor()])):
        """
        Dataset adapter for the GT4HistOCR data set

        :param dset_path: path to the "corpus" folder of the dataset, containing the corpora
        :param corpora: list of Corpus-Enum variables, indicating which corpora to use (defaults to all corpora)
        :param img_shape: shape of the images
        """
        # gather image paths and gt annotations
        img_paths, gt_paths = map(np.array, search_files(dset_path, suffixes=['.png',  '.txt']))
        # filter for used corpora
        fltr_in_corpora = (np.sum([np.char.count(img_paths, c.value) for c in corpora], axis=0) > 0).astype(bool)
        img_paths,  gt_paths = img_paths[fltr_in_corpora], gt_paths[fltr_in_corpora]
        # initialize the OCR data set
        alphabet = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GT4HistOCR_alphabet.npy'))
        super().__init__(img_paths,  gt_paths, f_split=f_split, transformation=transformation, alphabet=alphabet)

