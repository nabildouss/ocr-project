import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))), '..')
from src import ctc_decoder, baseline, clstm_eval, clstm_train, evaluation, data
import src.milestone1 as ms1
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import pickle
import tqdm


def torch_confidence(model, dset, prog_bar=True, s_batch=4, n_workers=4):
    if prog_bar:
        prog_bar = tqdm.tqdm(total=len(dset))
    # the training loop
    it_count = 0
    L_IN = [int(model.sequence_length) for _ in range(s_batch)]
    # new data loader initialization required after each epoch
    dloader = DataLoader(dset, batch_size=s_batch, num_workers=n_workers, shuffle=True,
                         collate_fn=dset.batch_transform)
    confidences = []
    predictions = []
    targets = []
    for batch, tgt, l_targets in dloader:
        y = model(batch)
        pred, conf = ctc_decoder.torch_confidence(log_P=y)
        confidences.append(*conf)
        predictions.append(*pred)
        targets.append(*tgt)
    return predictions, confidences, targets


def parser_torch():
    return evaluation.arg_parser()


def sw(data_set, corpora, pixels, pth_model, seq_len=256, prog_bar=True):
    _, test = ms1.load_data(data_set,
                            transformation=Compose([Resize([pixels, pixels * seq_len]), ToTensor()]),
                            corpora=corpora, cluster=True)
    # setting up the (baseline-) model
    model = baseline.BaseLine3(n_char_class=len(test.character_classes) + 1, shape_in=(1, pixels, pixels * seq_len),
                               sequence_length=seq_len)
    # loading the model
    state_dict = torch.load(pth_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return torch_confidence(model, dset=test, prog_bar=prog_bar)


def main_method(mode='torch'):
    if mode == 'torch':
        ap = parser_torch().parse_args()
        if ap.model_type == 'Baseline3':
            preds, confs, targets = sw(data_set=ap.data_set, corpora=[data.ALL_CORPORA[ap.corpus_ids]], pixels=32,
                                       pth_model=ap.pth_model, prog_bar=ap.prog_bar)
        else:
            raise ValueError(f'unknown model: {ap.model_type}')
        if not os.path.isdir(ap.out):
            os.makedirs(ap.out)
        with open(os.path.join(ap.out, 'predictions.pkl'), 'wb') as f_pred:
            pickle.dump(predictions, f_pred)
        with open(os.path.join(ap.out, 'confidences.pkl'), 'wb') as f_conf:
            pickle.dump(confidences, f_conf)
        with open(os.path.join(ap.out, 'targets.pkl'), 'wb') as f_tgt:
            pickle.dump(targets, f_tgt)
        return preds, confs, targets


if __name__ == '__main__':
    predictions, confidences = main_method('torch')
    # predictions, confidences = main_method('clstm')
