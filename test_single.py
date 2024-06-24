import os
import cv2
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import *
from CustomDataset import *
from ConvNet import *

import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = f'./model/CustomConvNet/trial_1/'

test_dataset = torch.load('./test_dataset_pytorch.pth')
test_data = list()
for idx in range(len(test_dataset)):
    test_data.append(test_dataset[idx])
test_data = np.array(test_data)

test_data = torch.tensor(test_data, dtype = torch.float32) / 255.0

# model = CustomInceptionResNetSingle(1, 512, 140)
# model.setup([32, 1, 28, 28])
# model = model.to(device)

model = CustomConvNet(1, 140)
model.setup([32, 1, 28, 28])
model = model.to(device)

msg = model.load_state_dict(torch.load(MODEL_PATH + f'Epoch_186_best_model.pth'))
print(msg)

model.eval()

pred_list = list()
with torch.no_grad():
    for image in tqdm(test_data, desc = 'Inferencing...'):
        image = image.to(device).unsqueeze(0).unsqueeze(0)
        pred = model(image)

        pred_list.append(pred.argmax(dim = 1).item())

print(pred_list)

df = pd.read_csv('test_dataset_wo_labels.csv')
df['class'] = pred_list

now = datetime.datetime.now()
df.to_csv(f'./20226744_박제현_{now.day}_{now.hour}_{now.minute}_{now.second}.csv', index = False)