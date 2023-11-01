import os
import sys
import idx2numpy
import torch
from torch.utils.data import Dataset


class Mnist(Dataset):
    def __init__(self, mode, path, transform=None, target_transform=None):
        self.path = path

        self.train_img = 'train-images.idx3-ubyte'
        self.test_img = 't10k-images.idx3-ubyte'
        self.train_label = 'train-labels.idx1-ubyte'
        self.test_label = 't10k-labels.idx1-ubyte'
        if mode == 'train':
            self.data_img = idx2numpy.convert_from_file(os.path.join(self.path, self.train_img))
            self.data_label = idx2numpy.convert_from_file(os.path.join(self.path, self.train_label))

        elif mode == 'test':
            self.data_img = idx2numpy.convert_from_file(os.path.join(self.path, self.test_img))
            self.data_label = idx2numpy.convert_from_file(os.path.join(self.path, self.test_label))

        else:
            sys.exit(0)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, idx):
        img = self.data_img[idx]
        label = self.data_label[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {'img' : torch.tensor(img, dtype=torch.float32), 'label' : label}

        return sample
