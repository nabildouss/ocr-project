import torch
from torchvision.transforms import Resize
import cv2
import numpy as np


class SlidingWindow:

    def __init__(self, seq_len=150):
        self.seq_len = seq_len

    def split_img(self, img):
        if len(img.shape) != 3:
            raise ValueError(f'img should have 3 dimensions, first dim for channel, second for rows, third for columns')
        if not img.shape[2] % self.seq_len == 0:
            raise ValueError(f'The image has the length {img.shape[1]}, which is not divisible by the sequence length {self.seq_len}')
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


class AutomaticWindow:

    def __init__(self, window_size=(32, 32)):
        self.window_size = window_size
        self.resize = Resize(self.window_size)

    def sliding_windows(self, img, thresh=0.5):
        bin = img > thresh
        contours, hierarchy = cv2.findContours(bin[0].detach().cpu().numpy().astype(np.uint8),
                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sarching for characters
        char_splits = [[np.min(c[:,:,0]), np.max(c[:, :, 0])] for c in contours]
        char_splits = self.__clean_up_overlaps(char_splits, img)
        # filling in blank spaces
        space_length = np.mean([s[1]-s[0] for s in char_splits]) / 3
        splits = []
        for i, s in enumerate(char_splits):
            splits.append(s)
            if i+1 < len(char_splits):
                if char_splits[i+1][0] - s[1] > space_length:
                    splits.append([s[1], char_splits[i+1][0]])
        # gather windows
        windows = []
        for s in splits:
            windows.append(img[:, :, s[0]:s[1]])
        return torch.stack([self.resize(w) for w in windows])

    def __clean_up_overlaps(self, splits, img=None):
        # helper functions
        def overlaps(a, b):
            if max(a) < min(b) or min(a) > max(b):
                return False
            a_vol = a[1] - b[0]
            b_vol = b[1] - b[0]
            intersection = max(a[0], b[0]), min(a[1], b[1])
            intersection_vol = intersection[1] - intersection[0]
            return intersection_vol / min(a_vol, b_vol) > 0.5
        def merge_splits(s):
            s = np.array(s)
            return np.min(s[:, 0]), np.max(s[:, 1])
        # finding out which splits overlap and grouping them accordingly
        splits = sorted(splits, key=lambda s: s[0])
        N = len(splits)
        group = [0]
        groups = []
        for i in range(N):
            if i == N-1 and group != []:
                groups.append(group)
            elif any([overlaps(splits[g], splits[i+1]) for g in group]):
                group.append(i+1)
            else:
                groups.append(group)
                group = [i+1]
        # finally merge the splits
        merged_splits = [merge_splits([splits[i] for i in m]) for m in groups]
        merged_splits = sorted(merged_splits, key=lambda x: x[0])
        return merged_splits

