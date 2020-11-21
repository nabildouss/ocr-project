import torch


class SlidingWindow:

    def __init__(self, seq_len=150):
        self.seq_len = seq_len

    def split_img(self, img):
        if len(img.shape) != 3:
            raise ValueError(f'img should have 3 dimensions, first dim for channel, second for rows, third for columns')
        if not img.shape[2] % self.seq_len == 0:
            raise ValueError(f'The image has the length {img.shape[1]}, which is not divisible by the sequence length {seq_len}')
        stride = int(img.shape[2] / self.seq_len)
        return torch.stack([img[:, :, c*stride:c*stride+stride] for c in range(self.seq_len)])

    def sliding_windows(self, img):
        splits = self.split_img(img)
        windows = []
        for i in range(self.seq_len):
            if i == 0:
                w = torch.cat([*splits[:3]], dim=2)
            elif i >= self.seq_len-3:
                w = torch.cat([*splits[-3:]], dim=2)
            else:
                w = torch.cat([*splits[i:i+3]], dim=2)
            windows.append(w)
        return torch.stack(windows)

