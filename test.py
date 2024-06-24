import os
import cv2
import pickle
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = f'./model/CustomMultiConvNet/trial_3/'

test_dataset = torch.load('./test_dataset_pytorch.pth')
test_data = list()
for idx in range(len(test_dataset)):
    test_data.append(test_dataset[idx])
test_data = np.array(test_data)

test_data = torch.tensor(test_data, dtype = torch.float32) / 255.0

model = CustomMultiConvNet(1, (14, 10), HIDDEN_SIZE = 512)
model.setup([32, 1, 28, 28])
model = model.to(device)

msg = model.load_state_dict(torch.load(MODEL_PATH + 'Epoch_76_best_model.pth'))
print(msg)

model.eval()

pred_list = list()
with torch.no_grad():
    model.eval()
    for data in tqdm(test_data):
        data = torch.tensor(data, dtype=torch.float32).view(1, 1, 28, 28).to(device)
        data = data / data.max()

        j_pred, m_pred = model(data)
        j_pred = j_pred.cpu().detach().numpy().argmax(axis=1)[0]
        m_pred = m_pred.cpu().detach().numpy().argmax(axis=1)[0]

        pred_list.append(j_pred * 10 + m_pred)
print(pred_list)

df = pd.read_csv('test_dataset_wo_labels.csv')
df['class'] = pred_list

now = datetime.datetime.now()
df.to_csv(f'./20226744_박제현_{now.day}_{now.hour}_{now.minute}_{now.second}.csv', index = False)


