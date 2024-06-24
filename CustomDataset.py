import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

class CustomTrainDataset(Dataset):
    def __init__(self, images, labels, n_classes = 140):
        self.images = images.reshape(-1, 1, 28, 28)
        self.labels = labels

        self.images = (self.images / self.images.max()).astype(np.float32)
        self.labels = np.eye(140)[self.labels].astype(np.float32)

        self.images = torch.tensor(self.images)
        self.labels = torch.tensor(self.labels)

        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class CustomMultiTrainDataset(Dataset):
    def __init__(self, images, j_labels, m_labels, j_classes = 14, m_classes = 10):
        self.images = images.reshape(-1, 1, 28, 28)
        self.j_labels = j_labels
        self.m_labels = m_labels

        self.images = (self.images / self.images.max()).astype(np.float32)
        self.j_labels = np.eye(14)[self.j_labels].astype(np.float32)
        self.m_labels = np.eye(10)[self.m_labels].astype(np.float32)

        self.images = torch.tensor(self.images)
        self.j_labels = torch.tensor(self.j_labels)
        self.m_labels = torch.tensor(self.m_labels)

        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.j_labels[idx], self.m_labels[idx]

class CustomTestDataset(Dataset):
    def __init__(self, images):
        self.images = images.reshape(-1, 1, 28, 28)

        self.images = (self.images / self.images.max()).astype(np.float32)
        self.images = torch.tensor(self.images)

        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images
        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]