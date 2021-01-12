import clstm
import os
from argparse import ArgumentParser


def save(model, file_name):
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    clstm.save_net(file_name, model)

def load(file_name):
    clstm.load_net(file_name)

def run_training(net, dataset, iterations=4e1):
    prog_bar = tqdm.tqdm(total=iterations)
    it_count = 0
    # TODO: dloader to clstm framework conversion
    while it_count < iterations:
        for batch, targets, l_targets in dloader:
            # gather data

            pass
            # forward pass
            pass
            # optimization step
            pass
            # breaking when iterations have been exceeded
            it_count += 1
            if not it_count < iterations:
                break


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--corpora_id', default=None)
    ap.add_argument('--out', default='CLSTM_models/my_model.clstm')
    ap.parse_args()

    corpora = [ALL_CORPORA[ap.corpora_id]]
    train, _ = ms1.load_data('GT4HistOCR',
                             transformation=Compose([Resize([32, 256]), ToTensor()]),
                             corpora=corpora)
    net = clstm.make_net_init("bidi", f"ninput=32:nhidden=100:noutput={len(train.character_classes)+1}")
    net.setLearningRate(1e-2, 0.9)#(1e-4, 0.9)

    run_train(net, train)
    save(net, ap.out)
