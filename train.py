import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils import *
from CustomDataset import *
from ConvNet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_PATH = './DLT_entity/generated_train/'
VALID_PATH = './DLT_entity/generated_valid/'
MODEL_PATH = './model/CustomMultiConvNet/trial_7/'
FILE_PATH = './font_texts/'

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

LAMBDA_J = 14 / 10

images, labels, j_labels, m_labels = list(), list(), list(), list()

for label in tqdm(os.listdir(FILE_PATH), desc = 'Loading Data...'):
    label = int(label)
    j_label = label // 10
    m_label = label % 10
    for filename in os.listdir(FILE_PATH + f'{label}'):
        image = cv2.imread(FILE_PATH + f'{label}' + f'/{filename}')[:, :, 0]
        image = (image / image.max()) if image.mean() > 0.0 else image

        images.append(image)
        labels.append(label)
        j_labels.append(j_label)
        m_labels.append(m_label)

images = np.array(images)
labels = np.array(labels)
j_labels = np.array(j_labels)
m_labels = np.array(m_labels)

train_images, valid_images, train_j_labels, valid_j_labels, train_m_labels, valid_m_labels = train_test_split(images, j_labels, m_labels, test_size = 0.3, shuffle = True)

train_dataset = CustomMultiTrainDataset(train_images, train_j_labels, train_m_labels)
valid_dataset = CustomMultiTrainDataset(valid_images, valid_j_labels, valid_m_labels)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = True)

# model = CustomInceptionResNetSingle(1, 512, 140)
# model.setup(next(iter(train_loader))[0].shape)
# model = model.to(device)

model = CustomMultiConvNet(1, (14, 10), HIDDEN_SIZE = 512)
model.setup(next(iter(train_loader))[0].shape)
model = model.to(device)

j_criterion = nn.CrossEntropyLoss().to(device)
m_criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_loss = list()
valid_loss = list()
train_accuracy = list()
valid_accuracy = list()

best_score = np.inf

for epoch in range(EPOCHS):
    model.train()
    running_loss = float()
    running_accuracy = float()
    for images, j_labels, m_labels in tqdm(train_loader, desc = f'Epoch {epoch} Training...'):
        images, j_labels, m_labels = images.to(device), j_labels.to(device), m_labels.to(device)

        j_pred, m_pred = model(images)
        j_loss = j_criterion(j_pred, j_labels)
        m_loss = m_criterion(m_pred, m_labels)
        loss = (LAMBDA_J * j_loss + m_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += (get_accuracy(j_pred, j_labels) + get_accuracy(m_pred, m_labels)) / 2

    train_loss.append(running_loss / len(train_loader))
    train_accuracy.append(running_accuracy / len(train_loader))

    model.eval()
    running_loss = float()
    running_accuracy = float()
    with torch.no_grad():
        for images, j_labels, m_labels in tqdm(valid_loader, desc=f'Epoch {epoch} Validating...'):
            images, j_labels, m_labels = images.to(device), j_labels.to(device), m_labels.to(device)

            j_pred, m_pred = model(images)
            j_loss = j_criterion(j_pred, j_labels)
            m_loss = m_criterion(m_pred, m_labels)
            loss = (LAMBDA_J * j_loss + m_loss) / 2

            running_loss += loss.item()
            running_accuracy += (get_accuracy(j_pred, j_labels) + get_accuracy(m_pred, m_labels)) / 2

    valid_loss.append(running_loss / len(valid_loader))
    valid_accuracy.append(running_accuracy / len(valid_loader))

    if best_score > valid_loss[-1]:
        print(f'Lowest Validation Loss Renewed! {best_score:.3f} -> {valid_loss[-1]:.3f}')
        torch.save(model.state_dict(), MODEL_PATH + f'Epoch_{epoch}_best_model.pth')
        best_score = valid_loss[-1]

    print(f'Epoch {epoch} | Train Loss: {train_loss[-1]:.3f}, Train Accuracy: {train_accuracy[-1]:.3f} | Valid Loss: {valid_loss[-1]:.3f}, Valid Accuracy: {valid_accuracy[-1]:.3f}')

plt.figure(figsize = (12, 7))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label = 'Train Loss')
plt.plot(valid_loss, label = 'Valid Loss')
plt.title(f'Loss')
plt.xlabel(f'Epochs')
plt.ylabel(f'Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label = 'Train Accuracy')
plt.plot(valid_accuracy, label = 'Valid Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()
