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

FILE_PATH = './font_texts/'
MODEL_PATH = './model/CustomConvNet/trial_1/'

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

text_to_idx = dict(zip(text_list, list(range(140))))
idx_to_text = dict(zip(list(range(140)), text_list))

images, labels = list(), list()

for label in tqdm(os.listdir(FILE_PATH), desc = 'Loading Data...'):
    label = int(label)
    for filename in os.listdir(FILE_PATH + f'{label}'):
        image = cv2.imread(FILE_PATH + f'{label}' + f'/{filename}')[:, :, 0]
        image = (image / image.max()) if image.mean() > 0.0 else image

        images.append(image)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, shuffle = True, stratify = labels)

train_dataset = CustomTrainDataset(train_images, train_labels)
valid_dataset = CustomTrainDataset(valid_images, valid_labels)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = True)

dummy_batch = next(iter(train_loader))
dummy_images = dummy_batch[0]
dummy_labels = dummy_batch[1]
plt.figure(figsize = (20, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(dummy_images[i][0])
    plt.title(f'Min: {dummy_images[i].min()}\nMax: {dummy_images[i].max()}\nlabel: {dummy_labels[i].argmax().item()}')

plt.show()

model = CustomConvNet(1, 140)
model.setup(next(iter(train_loader))[0].shape)
model = model.to(device)

print(model.load_state_dict(torch.load(MODEL_PATH + f'Epoch_99_best_model.pth')))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_loss = list()
valid_loss = list()
train_accuracy = list()
valid_accuracy = list()

best_score = 4.132

for epoch in range(100, 100 + EPOCHS):
    model.train()
    running_loss = float()
    running_accuracy = float()
    for images, labels in tqdm(train_loader, desc = f'Epoch {epoch} Training...'):
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += get_accuracy(pred, labels)

    train_loss.append(running_loss / len(train_loader))
    train_accuracy.append(running_accuracy / len(train_loader))

    model.eval()
    running_loss = float()
    running_accuracy = float()
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f'Epoch {epoch} Validating...'):
            images, labels = images.to(device), labels.to(device)

            pred = model(images)
            loss = criterion(pred, labels)

            running_loss += loss.item()
            running_accuracy += get_accuracy(pred, labels)

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
