import os
import cv2
import pickle
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import *
from CustomDataset import *
from ConvNet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = torch.load('./test_dataset_pytorch.pth')
test_data = list()
for idx in range(len(test_dataset)):
    test_data.append(test_dataset[idx])
test_data = np.array(test_data)

test_data = torch.tensor(test_data, dtype = torch.float32) / 255.0

model_list = list()
for i, epoch in zip(range(6), [96, 97, 76, 93, 91, 97]):
    model = CustomMultiConvNet(1, (14, 10), HIDDEN_SIZE = 512)
    model.setup([32, 1, 28, 28])
    model = model.to(device)
    print(model.load_state_dict(torch.load(f'./model/CustomMultiConvNet/trial_{i + 1}/' + f'Epoch_{epoch}_best_model.pth')))

    model_list.append(model)

pred_list = list()
with torch.no_grad():
    for data in tqdm(test_data):
        j_prob, m_prob = None, None
        for model in model_list:

            model.eval()
            data = torch.tensor(data, dtype=torch.float32).view(1, 1, 28, 28).to(device)
            data = data / data.max()

            j_pred, m_pred = model(data)
            j_pred = j_pred.cpu().detach().numpy()
            m_pred = m_pred.cpu().detach().numpy()

            j_prob = j_pred if type(j_prob) == type(None) else (j_prob + j_pred)
            m_prob = m_pred if type(m_prob) == type(None) else (m_prob + m_pred)

        j = j_prob.argmax(axis = 1)[0]
        m = m_prob.argmax(axis = 1)[0]

        pred_list.append(j * 10 + m)

print(pred_list)

df = pd.read_csv('test_dataset_wo_labels.csv')
df['class'] = pred_list

now = datetime.datetime.now()
df.to_csv(f'./20226744_박제현_{now.day}_{now.hour}_{now.minute}_{now.second}.csv', index = False)


