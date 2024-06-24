# Code for data load (Pytorch version)
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load test data name
uuid_df = pd.read_csv('test_dataset_wo_labels.csv')
uuid_mapping = uuid_df['obj_name'].tolist()

class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images
        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

dataset_path = './test_dataset_pytorch.pth' #Modify this path to your environments
test_dataset = torch.load(dataset_path)

test_data = list()
for idx in range(len(test_dataset)):
    test_data.append(test_dataset[idx])

test_data = torch.tensor(test_data)

print(test_data.shape)

import matplotlib.pyplot as plt
for image in test_data[0:10]:
    plt.imshow(image)
    plt.show()