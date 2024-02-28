# This is updated from HAMimageclassification.py to include weights to balance the data

'''
The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import random
import torch.nn.functional as F # softmax
from tqdm import tqdm # progress visualisation

import torch
import torchvision
import torchvision.transforms as T # data augmentation
import torchvision.models as models # to get pretrained models
import torch.nn as nn # to build NN, criterion
import torch.optim as optim # optimizer

# plotting and evaluation
from sklearn.metrics import confusion_matrix # performance evaluation

import pandas as pd # read csv
from imblearn.over_sampling import RandomOverSampler as ROS # training data oversampling
from sklearn.model_selection import train_test_split # splitting dataframes
from torch.utils.data import Dataset, DataLoader # data pipeline

# setting gpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For reproducibility
RANDOM_SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(RANDOM_SEED)

# Read HAM10000_metadata.csv
metadata = pd.read_csv('C:/Users/Kyra/Documents/GitHub/HAMImageClassification/HAM10000_metadata.csv')

# Read all images in HAM10000_images directory
image_dir = 'C:/Users/Kyra/Documents/GitHub/HAMImageClassification/HAM10000_images'
image_files = glob(os.path.join(image_dir, '*.jpg'))

# Label each image with its "dx" from HAM10000_metadata.csv
image_labels = []
image_ids = []
for image_file in image_files:
    image_id = os.path.splitext(os.path.basename(image_file))[0]
    dx = metadata.loc[metadata['image_id'] == image_id, 'dx'].values[0]
    image_labels.append(dx)
    image_ids.append(image_id + '.jpg')

# Create a dictionary to map unique labels to numbers
label_to_number = {label: number for number, label in enumerate(set(image_labels))}

# Convert image_labels to numbers using the dictionary
image_labels = [label_to_number[label] for label in image_labels]

label_counts = {}
for label in image_labels:
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1

# Print the number of images associated with each number label
for label, count in label_counts.items():
    print(f"Number of images with label {label}: {count}")

# Print the number of images and their corresponding labels
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(image_labels)}")


# Load the DeiT model
import timm
model = timm.create_model('deit_small_patch16_224', pretrained=True)

# Replace the classifier with a new one, that has 7 classes
model.head = nn.Linear(in_features=384, out_features=7, bias=True)

# Move the model to the GPU
model = model.to(DEVICE)

# Define the loss function and the optimizer
class_weights = [1 / label_counts[label] for label in range(len(label_counts))]
class_weights = torch.FloatTensor(class_weights).to(DEVICE)
print("Label numbers associated with each class weight:")
for label, weight in zip(range(len(label_counts)), class_weights):
    print(f"Label {label}: {weight.item()}")
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the data augmentation
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset
class HAM10000Dataset(Dataset):
    def __init__(self, image_ids, image_labels, image_dir, transform=None):
        self.image_ids = image_ids
        self.image_labels = image_labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_file = os.path.join(self.image_dir, image_id)
        image = Image.open(image_file)
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Split the data into training and validation sets
train_image_ids, val_image_ids, train_image_labels, val_image_labels = train_test_split(image_ids, image_labels, test_size=0.2, stratify=image_labels, random_state=RANDOM_SEED)

# Create the training and validation datasets
train_dataset = HAM10000Dataset(train_image_ids, train_image_labels, image_dir, transform=transform)

val_dataset = HAM10000Dataset(val_image_ids, val_image_labels, image_dir, transform=transform)

# Create the training and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {running_loss/len(train_dataloader)}")

# Evaluate the model
model.eval()
running_loss = 0.0
running_corrects = 0
for images, labels in tqdm(val_dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(images)
    loss = criterion(outputs, labels)
    running_loss += loss.item()
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels)

print(f"Loss: {running_loss/len(val_dataloader)}, Accuracy: {running_corrects.float()/len(val_dataset)}")

# Plot the confusion matrix
model.eval()
y_true = []
y_pred = []
for images, labels in tqdm(val_dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.matshow(cm, cmap='gray')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
torch.save(model.state_dict(), 'model.pth')
