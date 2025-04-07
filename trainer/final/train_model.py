import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import re
import json
from sklearn.model_selection import train_test_split
import jiwer
import matplotlib.pyplot as plt
from config.config import annotations_file, img_dir, idx2word_path, max_lengths, num_epochs, model_encode_path, model_decode_path
from .training_utilities import word2idx, evaluate_model



# Encoding and decoding functions
def encode_khmer_word(word, word2idx):
    indices = []
    for character in word:
        index = word2idx.get(character)
        if index is None:
            return None, f"Character '{character}' not found in vocabulary!"
        indices.append(index)
    return indices, None

def decode_indices(indices, idx2word):
    characters = []
    for index in indices:
        character = idx2word.get(str(index))
        if character is None:
            return None, f"Index '{index}' not found in idx2word!"
        characters.append(character)
    return ''.join(characters), None

# Model Definitions (EncoderCNN and DecoderRNN)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(hidden_size, hidden_size)  # Initialize LSTM hidden state
        self.init_c = nn.Linear(hidden_size, hidden_size)  # Initialize LSTM cell state

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(lstm_out)
        return outputs

# Image Captioning Dataset
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, img_labels, img_dir, vocab, transform=None, max_length=50):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        caption = self.img_labels.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        indices, error = encode_khmer_word(caption, self.vocab)
        if error:
            print(f"Error encoding caption: {error}")
            indices = [self.vocab['<UNK>']] * self.max_length
        tokens = [self.vocab['<START>']] + indices + [self.vocab['<END>']]
        tokens += [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        return image, torch.tensor(tokens[:self.max_length])

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load dataset
all_images = pd.read_csv(annotations_file, delimiter=' ', names=['image', 'caption'])

# Split dataset
train_images, eval_images, train_captions, eval_captions = train_test_split(
    all_images['image'].tolist(), all_images['caption'].tolist(), test_size=0.2, random_state=42
)

train_dataset = ImageCaptionDataset(
    img_labels=pd.DataFrame({'image': train_images, 'caption': train_captions}),
    img_dir=img_dir,
    vocab=word2idx,
    transform=transform,
    max_length=max_lengths
)

eval_dataset = ImageCaptionDataset(
    img_labels=pd.DataFrame({'image': eval_images, 'caption': eval_captions}),
    img_dir=img_dir,
    vocab=word2idx,
    transform=transform,
    max_length=max_lengths
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
encoder = EncoderCNN(embed_size=512).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx), num_layers=1).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

# Training Loop
best_wer = float('inf')
 
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        loss = criterion(outputs.view(-1, len(word2idx)), captions[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader)}')
    _, wer = evaluate_model(encoder, decoder, eval_loader, device, epoch)

    if wer < best_wer:
        best_wer = wer

# Save the encoder and decoder models
torch.save(encoder.state_dict(), model_encode_path)
torch.save(decoder.state_dict(), model_decode_path)
