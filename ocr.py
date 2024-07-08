import os
import urllib.request
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = './data'
BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Download EMNIST dataset using torchvision
def download_emnist(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'emnist')):
        os.makedirs(os.path.join(data_dir, 'emnist'))
        print('Downloading EMNIST dataset...')
        datasets.EMNIST(root=os.path.join(data_dir, 'emnist'), split='byclass', download=True)
        print('EMNIST dataset downloaded.')
    else:
        print('EMNIST dataset already exists.')

# Preprocess EMNIST dataset
def emnist_to_images(data_dir, output_dir, split_ratio=0.8):
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    emnist_train = datasets.EMNIST(root=os.path.join(data_dir, 'emnist'), split='byclass', train=True, download=False)
    emnist_test = datasets.EMNIST(root=os.path.join(data_dir, 'emnist'), split='byclass', train=False, download=False)
    
    def save_images_and_labels(dataset, output_dir):
        for i, (image, label) in enumerate(tqdm(dataset)):
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
            image = image.resize((128, 32))
            image_path = os.path.join(output_dir, f'{i}.png')
            label_path = os.path.join(output_dir, f'{i}.txt')
            image.save(image_path)
            with open(label_path, 'w') as f:
                f.write(str(label.item()))
    
    train_count = int(len(emnist_train) * split_ratio)
    save_images_and_labels([(emnist_train.data[i], emnist_train.targets[i]) for i in range(train_count)], os.path.join(output_dir, 'train'))
    save_images_and_labels([(emnist_train.data[i], emnist_train.targets[i]) for i in range(train_count, len(emnist_train))], os.path.join(output_dir, 'test'))
    save_images_and_labels([(emnist_test.data[i], emnist_test.targets[i]) for i in range(len(emnist_test))], os.path.join(output_dir, 'test'))

# EMNIST Dataset class
class EMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.labels = [f.replace('.png', '.txt') for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])
        
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        with open(label_path, 'r') as f:
            label = f.read().strip()
        
        return image, label

# Data loader
def get_data_loaders(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = EMNISTDataset(root_dir=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = EMNISTDataset(root_dir=os.path.join(data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.rnn_input_size = 128 * 4
        self.rnn = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.permute(0, 3, 1, 2)  # NCHW to NHWC
        out = out.view(out.size(0), out.size(1), -1)
        
        out, _ = self.rnn(out)
        out = self.fc(out)
        
        return out

# Train the model
def train():
    # Load data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE, os.path.join(DATA_DIR, 'processed'))
    
    # Initialize model, loss function, and optimizer
    model = CNN(num_classes=62 + 1).to(device)
    criterion = nn.CTCLoss(blank=62, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = [text_to_indices(label) for label in labels]
            label_lengths = [len(label) for label in labels]
            labels = torch.tensor([item for sublist in labels for item in sublist], dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            outputs = outputs.permute(1, 0, 2)  # TxNxC
            
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
            
            # Compute CTC loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

# Evaluate the model
def evaluate():
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = EMNISTDataset(root_dir=os.path.join(DATA_DIR, 'processed', 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = CNN(num_classes=62 + 1).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    all_decoded_texts = []
    images = None
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            decoded_texts = decode_output(outputs)
            all_decoded_texts.extend(decoded_texts)
    
    # Save results
    with open('recognized_texts.txt', 'w') as f:
        for text in all_decoded_texts:
            f.write(text + '\n')
    
    # Display some results
    for i in range(5):
        img = images[i].cpu().numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f'Recognized Text: {all_decoded_texts[i]}')
        plt.show()

# Helper functions
def text_to_indices(text):
    return [char_to_idx[char] for char in text]

def decode_output(output):
    output = output.permute(1, 0, 2)  # TxNxC to NxTxC
    output = F.log_softmax(output, 2)
    output = torch.argmax(output, 2)
    output = output.cpu().numpy()

    decoded_texts = []
    for sequence in output:
        text = ""
        for char_idx in sequence:
            if char_idx != 62:  # Ignore the blank token
                text += idx_to_char[char_idx]
        decoded_texts.append(text)
    return decoded_texts

# Character set and mapping
characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main function to orchestrate the process
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='preprocess, train, or evaluate')
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        download_emnist(DATA_DIR)
        emnist_to_images(DATA_DIR, os.path.join(DATA_DIR, 'processed'))
    elif args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        print("Invalid mode. Use 'preprocess', 'train', or 'evaluate'.")

