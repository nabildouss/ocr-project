import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import ctc_decoder, baseline, evaluation, data, visualize
import src.milestone1 as ms1
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import pickle
import json
import cv2
import tqdm
from ctcdecode import CTCBeamDecoder


def run_confidence(model, dset, f_forward, prog_bar=True, s_batch=1, n_workers=4, beam_width=100, device=None):
    if prog_bar:
        prog_bar = tqdm.tqdm(total=len(dset))
    # the training loop
    # it_count = 0
    # L_IN = [int(model.sequence_length) for _ in range(s_batch)]
    # new data loader initialization required after each epoch
    dloader = DataLoader(dset, batch_size=s_batch, num_workers=n_workers, shuffle=False,
                         collate_fn=dset.batch_transform)
    confidences = []
    predictions = []
    targets = []
    lengths = []
    decoder = CTCBeamDecoder(
        ['{'] + [c for c in dset.character_classes],
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=beam_width,
        num_processes=4,
        blank_id=0,
        log_probs_input=True# if device is not None else False
    )
    i = 1
    if device is not None:
        model.eval()
        model = model.to(device)
    for batch, tgt, l_targets in dloader:
        #print(f'target:      {dset.embedding_to_word(tgt)}')
        batch = batch.to(device)
        y = f_forward(model, batch)
        pred, conf = ctc_decoder.torch_confidence(log_P=y, dset=dset, decoder=decoder)
        confidences.append(conf)
        predictions.append(pred)
        for len_tgt in l_targets:
            lengths.append(len_tgt)
        targets.append(tgt)
        if i % 100 == 0:
            print(f'pred: {dset.embedding_to_word(pred)}\n' +
                  f'gt:   {dset.embedding_to_word(tgt)}')
        i += 1
        prog_bar.update(1)
    confidences = np.array(confidences)
    return predictions, confidences, targets, lengths


def torch_confidence(model, dset, prog_bar=True, s_batch=1, n_workers=4, beam_width=100, device=None):
    return run_confidence(model=model, dset=dset, prog_bar=prog_bar, s_batch=s_batch, n_workers=n_workers,
                          beam_width=beam_width, device=device, f_forward=torch_forward)


def parser_torch():
    return evaluation.arg_parser()


def sw(data_set, corpora, pixels, pth_model, seq_len=256, prog_bar=True, cluster=True, device=None, beam_width=100):
    _, test = ms1.load_data(data_set,
                            transformation=Compose([Resize([pixels, pixels * seq_len]), ToTensor()]),
                            corpora=corpora, cluster=cluster)
    # setting up the (baseline-) model
    model = baseline.BaseLine3(n_char_class=len(test.character_classes) + 1, shape_in=(1, pixels, pixels * seq_len),
                               sequence_length=seq_len)
    # loading the model
    state_dict = torch.load(pth_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)
    model.eval()
    y_pred, p_conf, y, lengths = torch_confidence(model, dset=test, prog_bar=prog_bar, device=device, beam_width=beam_width)
    model.cpu()
    cer, wer = cer_wer(y_pred, y, test)

    sorted_err = np.argsort((cer + p_conf) ** 2)
    worst = [test[i] for i in sorted_err[-4:]]
    best = [test[i] for i in sorted_err[:4]]
    explanations_worst = []
    explanations_best = []
    for i in range(4):
        worst_imgs, worst_targets, worst_ltargets = test.batch_transform(worst[i:i + 1])
        explanations_worst.append([
            visualize.explanation_plot(worst_imgs, model, worst_targets,
                                        L_IN=[seq_len], l_targets=worst_ltargets, framework='torch'),
            worst_imgs
        ])
        best_imgs, best_targets, best_ltargets = test.batch_transform(best[i:i + 1])
        explanations_best.append([
            visualize.explanation_plot(best_imgs, model, best_targets,
                                       L_IN=[seq_len], l_targets=best_ltargets, framework='torch'),
            best_imgs
        ])
    explanations = [explanations_worst, explanations_best]
    return y_pred, p_conf, y, cer, wer, explanations, lengths


def torch_forward(model, batch):
    return model(batch).detach().cpu()
    

def clstm_forward(net, batch):
    x_in = batch.cpu().detach().numpy().reshape(*batch.shape[-2:]).T
    scale_factor = 32 / x_in.shape[1]
    x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
    x_in = x_in[:, :, None]
    # forward pass
    net.inputs.aset(x_in)
    net.forward()
    y_pred = net.outputs.array()
    y_pred = y_pred.transpose(0,2,1)
    return np.log(y_pred)


def clstm_confidence(net, dset, prog_bar=True, s_batch=1, n_workers=4, beam_width=100):
    return run_confidence(model=net, dset=dset, prog_bar=prog_bar, s_batch=s_batch, n_workers=n_workers,
                          beam_width=beam_width, f_forward=clstm_forward)
    # if prog_bar:
    #     prog_bar = tqdm.tqdm(total=len(dset))
    # # the training loop
    # # it_count = 0
    # # L_IN = [int(net.sequence_length) for _ in range(s_batch)]
    # # new data loader initialization required after each epoch
    # dloader = DataLoader(dset, batch_size=s_batch, num_workers=n_workers, shuffle=True,
    #                      collate_fn=dset.batch_transform)
    # confidences = []
    # predictions = []
    # targets = []
    # for batch, tgt, l_targets in dloader:
    #     x_in = batch.cpu().detach().numpy().reshape(*batch.shape[-2:]).T
    #     scale_factor = 32 / x_in.shape[1]
    #     x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
    #     x_in = x_in[:, :, None]
    #     # forward pass
    #     net.inputs.aset(x_in)
    #     net.forward()
    #     y_pred = net.outputs.array()
    #     # confidences
    #     pred, conf = ctc_decoder.torch_confidence(log_P=y_pred)
    #     # storing information
    #     confidences.append(*conf)
    #     predictions.append(*pred)
    #     targets.append(*tgt)
    #     # moving progress bar
    #     if prog_bar:
    #         prog_bar.update(1)
    # return predictions, confidences, targets
    

def cer_wer(predictions, targets, dset):
    cers = []
    wers = []
    for h, r in zip(predictions, targets):
        hyp, ref = map(dset.embedding_to_word, [h, r])
        cers.append(evaluation.adjusted_cer(reference=ref, hypothesis=hyp))
        wers.append(evaluation.adjusted_wer(reference=ref, hypothesis=hyp))
    cers = np.array(cers)
    wers = np.array(wers)
    return cers, wers


def parser_clstm():
    return clstm_eval.parser()


def clstm(data_set, corpora, pth_model, prog_bar=True, cluster=True, beam_width=100):
    _, test = ms1.load_data(data_set=data_set, corpora=corpora, cluster=cluster, transformation=Compose([ToTensor()]))
    # _, test = ms1.load_data(corpora=corpora, cluster=False, transformation=Compose([ToTensor()]))
    # construct network
    ninput = 32
    noutput = len(test.character_classes) + 1
    net = clstm_eval.load(pth_model)
    # gather confidences
    y_pred, p_conf, y, lengths = clstm_confidence(net=net, dset=test, prog_bar=prog_bar, beam_width=beam_width)
    cer, wer = cer_wer(y_pred, y, test)
    
    sorted_err = np.argsort((cer + p_conf)**2)
    worst = [test[i] for i in sorted_err[-4:]]
    best = [test[i] for i in sorted_err[:4]]
    explanations_worst = []
    explanations_best = []
    for i in range(4):
        worst_imgs, worst_targets, worst_ltargets = test.batch_transform(worst[i:i+1])
        explanations_worst.append([
            visualize.explanation_plot(worst_imgs, net, worst_targets,
                                       L_IN=[], l_targets=worst_ltargets, framework='clstm'),
            worst_imgs
        ])
        best_imgs, best_targets, best_ltargets = test.batch_transform(best[i:i + 1])
        explanations_best.append([
            visualize.explanation_plot(best_imgs, net, best_targets,
                                       L_IN=[], l_targets=best_ltargets, framework='clstm'),
            best_imgs
        ])
    explanations = [explanations_worst, explanations_best]
    return y_pred, p_conf, y, cer, wer, explanations, lengths



def main_method(mode='torch', cluster=True):
    if mode == 'torch':
        ap = parser_torch()
        ap.add_argument('--beam_width', default=1, type=int)
        ap = ap.parse_args()
        device = torch.device(ap.device)
        if ap.model_type == 'Baseline3':
            preds, confs, targets, cer, wer, explanations, lengths = sw(
                data_set=ap.data_set, corpora=[data.ALL_CORPORA[int(ap.corpus_ids)]], pixels=32,
                pth_model=ap.pth_model, prog_bar=ap.prog_bar, cluster=cluster, device=device,
                beam_width=ap.beam_width
            )
        else:
            raise ValueError(f'unknown model: {ap.model_type}')
    elif mode == 'clstm':
        ap = parser_clstm()
        ap.add_argument('--beam_width', default=1, type=int)
        ap = ap.parse_args()
        preds, confs, targets, cer, wer, explanations, lengths = clstm(
            data_set='GT4HistOCR', corpora=[data.ALL_CORPORA[ap.corpus_id]],
            pth_model=ap.clstm_path, prog_bar=ap.prog_bar, cluster=cluster,
            beam_width=ap.beam_width
        )
    else:
        raise ValueError(f'unknown mode: {mode}')
    write_results(ap.out, preds, confs, targets, list(cer), list(wer), explanations)
    visualize.confidence_plot(cer=cer, confs=confs, save_path=os.path.join(ap.out, 'conf_plot'))
    visualize.len_plot(cer=cer, lengths=lengths, save_path=os.path.join(ap.out, 'len_plot'))
    visualize.corrections_plot(err=cer, save_path=os.path.join(ap.out, 'corrections_plot_CER'))
    visualize.corrections_plot(err=wer, save_path=os.path.join(ap.out, 'corrections_plot_WER'))
    return preds, confs, targets, cer, wer


def write_results(out, preds, confs, targets, cer, wer, explanations=None):
        if not os.path.isdir(out):
            os.makedirs(out)
        with open(os.path.join(out, 'predictions.pkl'), 'wb') as f_pred:
            pickle.dump(preds, f_pred)
        with open(os.path.join(out, 'confidences.pkl'), 'wb') as f_conf:
            pickle.dump(confs, f_conf)
        with open(os.path.join(out, 'targets.pkl'), 'wb') as f_tgt:
            pickle.dump(targets, f_tgt)
        with open(os.path.join(out, 'WER_CER.json'), 'w') as f_cer:
            json.dump({'cer_list': cer, 'cer': np.mean(cer), 'wer_list': wer, 'wer':np.mean(wer)}, f_cer)
        with open(os.path.join(out, 'explanations.pkl'), 'wb') as f_explain:
            pickle.dump(explanations, f_explain)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    if os.path.isdir('../corpus'):
        y_pred, p_conf, y, cer, wer = main_method('torch', cluster=False)
    # from src import clstm_eval, clstm_train
    else:
        from src import clstm_eval
        main_method('clstm', cluster=True)
    
