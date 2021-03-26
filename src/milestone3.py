from src import ctc_decoder
import cv2


def predict(x, model, decoder, dset, framework='torch'):
    if framework == 'torch':
        log_P = model(x)
        pred, conf = ctc_decoder.torch_confidence(log_P=log_P, dset=dset, decoder=decoder)
        return pred, conf
    elif framework == 'clstm':
        x_in = x.cpu().detach().numpy().reshape(*x.shape[-2:]).T
        scale_factor = 32 / x_in.shape[1]
        x_in = cv2.resize(x_in, (32, int(x_in.shape[0] * scale_factor)))
        x_in = x_in[:, :, None]
        # forward pass
        model.inputs.aset(x_in)
        model.forward()
        log_P = model.outputs.array()
        pred, conf = ctc_decoder.torch_confidence(log_P=log_P, dset=dset, decoder=decoder)
        return pred, conf
    else:
        raise ValueError(f'unknown framework: {framework}\nPlease choose between torch or clstm')
