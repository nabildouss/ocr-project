import numpy as np
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.model import *
from src.data import *
import src.milestone1 as ms1


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def adjusted_wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case, delimiter)
    l_max = max(len(reference.split(delimiter)), len(hypothesis.split(delimiter)))
    return float(edit_distance) / l_max


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def adjusted_cer(reference, hypothesis, ignore_case=False, remove_space=False):
    edit_distance, _ = char_errors(reference, hypothesis, ignore_case, remove_space)
    l_max = max(len(reference), len(hypothesis))
    return float(edit_distance) / l_max


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def CTC_to_int(P):
    for p in P:
        pass
    pass


class Evaluator:

    def __init__(self, model, dset, device, s_batch=16, prog_bar=True):
        self.model = model
        self.dset = dset
        self.device = device
        self.s_batch = s_batch
        self.prog_bar = prog_bar

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        dloader = DataLoader(self.dset, batch_size=self.s_batch, num_workers=self.n_workers,
                             collate_fn=self.dset.batch_transform)
        if self.prog_bar:
            prog_bar = tqdm.tqdm(total=self.iterations)
        l_wer = []
        l_cer = []
        l_adj_wer = []
        l_adj_cer = []
        for batch, targets, l_targets in dloader:
            # moving the data to the (GPU-) device
            batch, targets = batch.to(self.device), targets.to(self.device)
            # forward pass
            y = self.model(batch)
            hypotheses = CTC_to_int(y)
            for h, r in zip(hypotheses, targets.cpu()):
                hyp, ref = map(self.dset.embedding_to_word, [h, r])
                l_wer.append(wer(ref, hyp))
                l_cer.append(cer(ref, hyp))
                l_adj_wer.append(adjusted_cer(ref, hyp))
                l_adj_cer.append(adjusted_cer(ref, hyp))
            if self.prog_bar:
                prog_bar.update(1)
        return map(np.mean, [l_wer, l_adj_wer, l_cer, l_adj_cer])


def arg_parser():
    ap = ArgumentParser()
    ap.add_argument('--data_set', default='GT4HistOCR', type=str)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--device', default='cpu', type=str)
    ap.add_argument('--prog_bar', default=True, type=bool)
    ap.add_argument('--out', default=None)
    ap.add_argument('--pth_model', default=None)
    return ap


def run_evaluation(pth_model, data_set, s_batch, device, prog_bar, pth_out):
    if pth_model is None:
        raise ValueError('Please submit a path to the model you want to evaluate')
    # loading the test split
    _, test = ms1.load_data(data_set, n_train=0.75, n_test=0.25)
    # loading the model
    state_dict = torch.load(pth_model, map_location=torch.device('cpu'))
    model = BaseLine(n_char_class=len(test.character_classes)+1)
    model.load_state_dict(state_dict=state_dict)
    # setting up the evaluation
    evaluator = Evaluator(model, test, device, s_batch=s_batch, prog_bar=prog_bar)
    # evaluating the model
    wer, adj_wer, cer, adj_cer = evaluator.eval()
    # setting up a dictionary to summariza evalutation
    summary = {'wer': wer, 'adj_wer': adj_wer, 'cer': cer, 'adj_cer': cer}
    # storing the dictionary as a JSON file
    if not  os.path.isdir(os.path.dirname(pth_out)):
        os.makedirs(os.path.dirname(pth_out))
    with open(pth_out, 'w') as f_out:
        json.dump(summary, f_out)
    # finally printing the results
    print(summary)


if __name__ ==  '__main__':
    ap = arg_parser().parse_args()
    run_evaluation(ap.pth_model, ap.data_set, ap.batch_size, ap.device, ap.prog_bar, ap.out)
