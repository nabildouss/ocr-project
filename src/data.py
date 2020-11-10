"""
This module provides classes and methods to load data sets, such as the GT4HistOCR data set.
All classes base torch.utils.data.Dataset, this way the torch.utils.data.DataLoader can be used.
Using the torch DataLoader is important, as we have a lazy loading approach and the DataLoader enables concurrent
workers in python that gather data for batches.
"""
from torchvision.transforms import ToTensor, Resize, Grayscale, Compose
from torch.utils.data import Dataset
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
    EarlyModernLatin = 1
    Kallimachos = 2
    RIDGES_Fraktur = 3
    RefCorpus_ENHG_Incunabula = 4
    dta19 = 5


# Most likely we will use all corpora, hence this global constant
ALL_CORPORA = tuple(Corpus)


# This dictionary maps the enum variables to the sudirectories in the "corpus" folder of GT4HistOCR
CORPUS_TO_SUBDIR = {Corpus.EarlyModernLatin: 'EarlyModernLatin', Corpus.Kallimachos: 'Kallimachos',
                    Corpus.RIDGES_Fraktur: 'RIDGES-Fraktur', Corpus.dta19: 'dta19',
                    Corpus.RefCorpus_ENHG_Incunabula: 'RefCorpus-ENHG-Incunabula'}


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

    def __init__(self, img_paths, gt_paths, f_split=None, transformation=ToTensor()):
        """
        :param img_paths: paths of all images
        :param gt_paths: paths to respective line transcriptions
        :param f_split: a function that filters out data from the img_paths and gt_paths, if None the data set stays whole
        :param img_shape: desired shape of images, if None no resizing takes place
        """
        super().__init__()
        self.img_paths = np.array(img_paths)
        self.gt_paths = np.array(gt_paths)
        # sorting to make splits deterministic
        sorted_idcs = np.argsort(self.img_paths)
        self.img_paths = self.img_paths[sorted_idcs]
        self.gt_paths = self.gt_paths[sorted_idcs]
        assert len(self.gt_paths) == len(self.img_paths), "Number of line image and and annotiation files have to be equal."
        self.trans = transformation
        self.grayscale = Grayscale(num_output_channels=1)
        self.f_split = f_split
        self.__apply_split()

    def __getitem__(self, idx):
        """
        :param idx: index of image NOTE: this is NOT the images ID, but simply it's files index in self.img_paths
        :return: Image tensor and transcription of the indexed line
        """
        return self.__load_img(self.img_paths[idx]), self.__line(self.gt_paths[idx])

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
        return tensor

    def __line(self, gt_path):
        """
        :param gt_path: path to the txt file containing the lines transcript
        :return: the lines transcript
        """
        with open(gt_path, 'r') as f_gt:
            line = ''.join(f_gt.readlines())
            line = line.rstrip()
        return line

    def __apply_split(self):
        """
        Applying the split function, filtering out paths
        """
        if self.f_split is not None:
            self.img_paths = self.f_split(self.img_paths)
            self.gt_paths = self.f_split(self.gt_paths)

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
        fltr_in_corpora = (np.sum([np.char.count(img_paths, CORPUS_TO_SUBDIR[c]) for c in corpora], axis=0) > 0).astype(bool)
        img_paths,  gt_paths = img_paths[fltr_in_corpora], gt_paths[fltr_in_corpora]
        # initialize the OCR data set
        super().__init__(img_paths,  gt_paths, f_split=f_split, transformation=transformation)

