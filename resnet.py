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
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import random
from tqdm import tqdm # progress visualisation

import torch
import torchvision
import torchvision.transforms as T # data augmentation
import torchvision.models as models # to get pretrained models
import torch.nn as nn # to build NN, criterion
import torch.optim as optim # optimizer
import time # to measure time

# plotting and evaluation
from sklearn.metrics import confusion_matrix # performance evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


import pandas as pd # read csv
from imblearn.over_sampling import RandomOverSampler as ROS # training data oversampling
from sklearn.model_selection import train_test_split # splitting dataframes
from torch.utils.data import Dataset, DataLoader # data pipeline

# Define the paths to the dataset and metadata CSV file
data_dir = 'C:/Users/Kyra/Documents/GitHub/HAMImageClassification/HAM10000_images'
metadata_file = 'C:/Users/Kyra/Documents/GitHub/HAMImageClassification/HAM10000_metadata.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load metadata from CSV file and create a mapping of 'dx' values to numbers
metadata = pd.read_csv(metadata_file)
label_to_number = {label: number for number, label in enumerate(metadata['dx'].unique())}

# Add a new column 'label' to metadata with numerical labels based on 'dx'
metadata['label'] = metadata['dx'].map(label_to_number)

# Create a folder for each class of images
for label in metadata['label'].unique():
    folder_path = os.path.join(data_dir, str(label))
    os.makedirs(folder_path, exist_ok=True)

# Move images to their respective folders based on the label
for index, row in metadata.iterrows():
    image_path = os.path.join(data_dir, row['image_id'] + '.jpg')
    label = row['label']
    destination_folder = os.path.join(data_dir, str(label))
    destination_path = os.path.join(destination_folder, row['image_id'] + '.jpg')

# Print the labels that correspond to each number
for label, number in label_to_number.items():
    print(f"Label: {label}, Number: {number}")

# Define data transformations and create DataLoader for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


# Load pre-trained ResNet model and modify the classifier for 7 classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Modify the fully connected layer for 7 classes

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
# Initialize the progress bar
progress_bar = tqdm(total=len(train_dataloader), desc='Epoch {}/{}'.format(epoch + 1, num_epochs))

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update the progress bar
        progress_bar.set_postfix({'Loss': running_loss / len(train_dataloader)})
        progress_bar.update()

    # Print the average loss for the epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}')

    # Reset the progress bar
    progress_bar.close()
    progress_bar = tqdm(total=len(train_dataloader), desc='Epoch {}/{}'.format(epoch + 2, num_epochs))

# Evaluate the model
model.eval()
val_loss = 0.0
val_correct = 0

with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()

val_loss /= len(val_dataloader)
val_accuracy = val_correct / len(val_dataset)

print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Save the trained model
torch.save(model.state_dict(), 'resnet_model_with_metadata.pth')
