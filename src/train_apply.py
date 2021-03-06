import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import training, evaluation


def train_apply(method='sw'):
    if method == 'sw':
        ap = training.main_method(model=method)
        ap.pth_model = ap.out
        ap.out = ap.out + '_eval'
        ap.model_type = 'Baseline3'
        preds = evaluation.main_method(ap)
        return preds
    elif method == 'lightweight':
        ap = training.main_method(model=method)
        ap.pth_model = ap.out
        ap.out = ap.out + '_eval'
        ap.model_type = 'Kraken'
        preds = evaluation.main_method(ap)
        return preds
    elif method == 'clstm':
        #clstm is only working on the cluster, hence the late import
        from src import clstm_eval, clstm_train
        ap = clstm_train.main_method()
        ap.clstm_path = ap.out
        ap.out = ap.out + '_eval'
        preds = clstm_eval.main_method(ap)
        return preds


if __name__ == '__main__':
    # NOTE: arg parsing is implicit either clstm_eval, clstm_tarain args parsing or training, evaluation args parsing

    # uncomment to use sliding window model
    # example: python3 train_apply.py --iterations=100000  --device=cpu  --batch_size=4 --out=models_100K_test/sw_0 --corpus_ids=0
    preds = train_apply('sw')

    # uncomment to use lightweight model
    # example: python3 train_apply.py --iterations=100000  --device=cpu  --batch_size=4 --out=models_100K_test/lw_0 --corpus_ids=0
    # preds = train_apply('lightweight')

    # uncomment to use CLSTM model
    # example: python3 train_apply.py --out=CLSTM_models_100K_test/clstm0.clstm --iterations=100000 --corpus_id=0
    # preds = train_apply('clstm')
