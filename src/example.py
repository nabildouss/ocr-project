import torch
from torch import nn
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class Kraken(nn.Module):

    def __init__(self, n_char_class=262):
        super().__init__()
        self.height = 48
        self.in_channels = 1
        self.n_char_class = n_char_class
        self.sequence_length = 150
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.lstm = nn.LSTM(input_size=64*12, hidden_size=self.n_char_class,
                            bidirectional=True)
        self.out = nn.Linear(2*self.n_char_class, self.n_char_class)

    def forward(self, x):
        assert x.shape[1] == self.in_channels and x.shape[2] == self.height
        y = self.conv1(x) # (N, 32, 48, 4*T)
        y = self.pool1(y) # (N, 32, 24, 2*T)
        y = self.conv2(y) # (N, 64, 24, 2*T)
        y = self.pool2(y) # (N, 64, 12, T)
        # reshaping
        bs = [y[:,:,:,i].flatten(start_dim=1) for i in range(y.shape[3])] # (T, N, 64*12)
        y = torch.stack(bs)
        # lstm
        y, _ = self.lstm(y)
        y = self.out(y)
        y = torch.log_softmax(y, dim=2)
        return y


class ToyData(nn.Module):

    def __init__(self, path):
        super().__init__()
        files = sorted(os.listdir(path))
        self.imgs = [os.path.join(path, f) for f in files if f.endswith('.png')]
        self.transcripts = [os.path.join(path, f) for f in files if f.endswith('.txt')]
        self.resize = Resize([48, 1024])
        self.trans = ToTensor()
        self.alphabet = ''
        for p in self.transcripts:
            self.alphabet += self.__load_trans(p)
        self.alphabet = sorted(set(self.alphabet))
        self.map_to_int = {c: i+1 for i, c in enumerate(self.alphabet)}
        self.map_to_char = {i+1: c for i, c in enumerate(self.alphabet)}

    def __getitem__(self, i):
        return self.__load_img(self.imgs[i]), self.__load_trans(self.transcripts[i])

    def __len__(self):
        return len(self.imgs)

    def __load_img(self, path):
        img = Image.open(path)
        tensor = self.trans(img)
        # grayscale conversion in case of RGB (3 channels) or RGB + opacity (4 channels) image representations
        if tensor.shape[0] > 1:
            tensor = self.grayscale(tensor[:3, :, :])
        # normalizae lighting
        tensor -= tensor.min()
        tensor /= tensor.max()
        # invert -> lines get propagated, not background
        tensor = 1 - tensor
        return self.resize(tensor)

    def __load_trans(self, path):
        with open(path, 'r') as f_trans:
            trans = ''.join(f_trans.readlines())
        return trans.strip()

    def grayscale(self, x):
        return rgb_to_grayscale(x, num_output_channels=1)

    def collate(self, data):
        batch, targets, l_targets = [], [], []
        for img, transcript in data:
            # images do not need to be changed
            batch.append(img)
            # mapping characters to integers
            emb = torch.tensor([self.map_to_int[c] for c in transcript])
            targets.append(emb)
            # keeping track of ooriginal lengths
            l_targets.append(len(emb))
        # Tensor conversion for batch and targets, the lengths can stay as a list
        # return torch.stack(batch), torch.nn.utils.rnn.pad_sequence(targets, batch_first=True), l_targets
        return torch.stack(batch), torch.cat(targets), l_targets


def to_str(idcs, dset):
    classes = [idcs[0]]
    for i in idcs[1:]:
        if i != classes[-1]:
            classes.append(i)
    return ''.join(dset.map_to_char[i.item()] for i in classes if i.item() != 0)


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
    dset = ToyData('toydata')
    model = Kraken(n_char_class=len(dset.alphabet)).to(device)
    # setup
    criterion = nn.CTCLoss(blank=0, reduction='mean').to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.002)
    # training loop
    dloader = DataLoader(dset, batch_size=4, num_workers=4, shuffle=True, collate_fn=dset.collate)
    it = 0
    while it < 10000:
        dloader = DataLoader(dset, batch_size=4, num_workers=4, shuffle=True, collate_fn=dset.collate)
        for img, emb, lens in dloader:
            y = model(img.to(device))
            loss = criterion(y, emb.to(device), [256 for _ in range(len(lens))], lens)
            loss.backward()
            optim.step()
            if it % 100 == 0:
                print(y[:, 0].shape)
                print(f'"{to_str(emb[:lens[0]], dset)}"')
                print(f'"{to_str(torch.argmax(y[:, 0], dim=1), dset)}"')
            it += 1
            if it >= 10000:
                break
