import os
import sys
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class Cifar10(Dataset):
    def __init__(self, mode, path, transform=None, target_transform=None):
        self.path = path

        if mode == 'train':
            self.data_img_folder = os.path.join(self.path, 'train')
            self.data_label_file = os.path.join(self.path, 'trainLabels.csv')
            self.data_label = pd.read_csv(self.data_label_file, names=['id', 'label'])

        elif mode == 'test':
            self.data_img_folder = os.path.join(self.path, 'train')
            self.data_label_file = os.path.join(self.path, 'sampleSubmission.csv')
            self.data_label = pd.read_csv(self.data_label_file, names=['id', 'label'])

        else:
            sys.exit(0)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_img_folder, self.data_label.iloc[idx, 0] + '.png')
        img = cv2.imread(img_path)
        label = self.data_label.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {'img' : torch.tensor(img, dtype=torch.float32), 'label' : label}

        return sample
