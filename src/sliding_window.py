import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import resize
import cv2
import numpy as np


class SlidingWindow:

    def __init__(self, seq_len=150):
        self.seq_len = seq_len

    def split_img(self, img):
        if len(img.shape) != 3:
            raise ValueError(f'img should have 3 dimensions, first dim for channel, second for rows, third for columns')
        if not img.shape[2] % self.seq_len == 0:
            raise ValueError(f'The image has the length {img.shape[2]}, which is not divisible by the sequence length {self.seq_len}')
        stride = int(img.shape[2] / self.seq_len)
        return torch.stack([img[:, :, c*stride:c*stride+stride] for c in range(self.seq_len)])

    def sliding_windows(self, img, n_windows=4):
        splits = self.split_img(img)
        windows = []
        for i in range(self.seq_len):
            if i == 0:
                w = torch.cat([*splits[:n_windows]], dim=2)
            elif i >= self.seq_len-n_windows:
                w = torch.cat([*splits[-n_windows:]], dim=2)
            else:
                w = torch.cat([*splits[i:i+n_windows]], dim=2)
            windows.append(w)
        return torch.stack(windows)


class PyramidalSlidingWindow(SlidingWindow):

    def __init__(self, seq_len=150):
        super().__init__(seq_len)

    def sliding_windows(self, img):
        # cutting out left end right void spaces
        size = img.shape[1:]
        cut_left = 0
        cut_right = img.shape[2]-1
        while img[:, :, cut_left].max() < 0.5:
            cut_left += 1
        while img[:, :, cut_right].max() < 0.5:
            cut_right -= 1
        img = img[:, :, cut_left:cut_right]
        img = resize(img, size=size)
        # setting up the pyramidal splits
        splits = self.split_img(img)
        inputs = []
        for i in range(self.seq_len):
            if i + 1 < self.seq_len and i - 1 > 0:
                left = splits[i-1]
                center = splits[i]
                right = splits[i+1]
                print(left.shape)
                empty = torch.zeros((*left.shape[:2], left.shape[2]*2))
                # 3 channels: left half, right half, botch halfes
                inputs.append(torch.stack([torch.cat([left, empty], dim=2),
                                           torch.cat([empty, right], dim=2),
                                           torch.cat([left, center, right], dim=2)]))
        return torch.stack(inputs)


class AutomaticWindow:

    def __init__(self, window_size=(32, 32)):
        self.window_size = window_size
        self.resize = Resize(self.window_size)

    def sliding_windows(self, img, thresh=0.8):
        bin = img > thresh
        contours, hierarchy = cv2.findContours(bin[0].detach().cpu().numpy().astype(np.uint8),
                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sarching for characters
        char_splits = [[np.min(c[:,:,0]), np.max(c[:, :, 0])] for c in contours]
        char_splits = [s for s in char_splits if s[1]-s[0]>0]
        char_splits = self.__clean_up_overlaps(char_splits, img)
        # filling in blank spaces
        space_length = np.mean([s[1]-s[0] for s in char_splits])
        splits = []
        for i, s in enumerate(char_splits):
            splits.append(s)
            if i+1 < len(char_splits):
                if char_splits[i+1][0] - s[1] > space_length:
                    print(f'space before {i + 1}th character')
                    splits.append([s[1], char_splits[i+1][0]])
        # gather windows
        windows = []
        for s in splits:
            windows.append(self.resize(img[:, :, s[0]:s[1]]))
        return torch.stack(windows)

    def __clean_up_overlaps(self, splits, img=None):
        # helper functions
        def overlaps(a, b):
            if max(a) < min(b) or min(a) > max(b):
                return False
            a_vol = a[1] - b[0]
            b_vol = b[1] - b[0]
            intersection = max(a[0], b[0]), min(a[1], b[1])
            intersection_vol = intersection[1] - intersection[0]
            return intersection_vol / min(a_vol, b_vol) > 0.25
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

