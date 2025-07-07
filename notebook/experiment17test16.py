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

# Function to load idx2word and convert it to word2idx
def load_vocabulary(path):
    with open(path, 'r') as file:
        idx2word = json.load(file)
    word2idx = {v: int(k) for k, v in idx2word.items()}
    return idx2word, word2idx

# Load vocabulary
idx2word_path = '/home/vitoupro/code/image_captioning/notebook/idx2word.json'
idx2word, word2idx = load_vocabulary(idx2word_path)

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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,dropout_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,dropout=dropout_prob if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(hidden_size, hidden_size)  # Initialize LSTM hidden state
        self.init_c = nn.Linear(hidden_size, hidden_size)  # Initialize LSTM cell state

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear(lstm_out)
        return outputs
    # def forward(self, features, captions, teacher_forcing_ratio=1.0):
    #     batch_size, seq_len = captions.size()
    #     vocab_size = self.linear.out_features

    #     # Store output predictions
    #     outputs = torch.zeros(batch_size, seq_len - 1, vocab_size).to(features.device)

    #     # Initialize hidden and cell state
    #     h = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
    #     c = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)

    #     # Start token as initial input
    #     input_token = self.dropout(self.embed(captions[:, 0])).unsqueeze(1)

    #     for t in range(1, seq_len):
    #         lstm_out, (h, c) = self.lstm(input_token, (h, c))
    #         lstm_out = self.dropout(lstm_out)
    #         output = self.linear(lstm_out.squeeze(1))  # (batch_size, vocab_size)
    #         outputs[:, t - 1, :] = output

    #         # Scheduled sampling decision
    #         teacher_force = torch.rand(1).item() < teacher_forcing_ratio
    #         next_input = captions[:, t] if teacher_force else output.argmax(1)
    #         input_token = self.dropout(self.embed(next_input)).unsqueeze(1)

    #     return outputs

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
annotations_file = '/home/vitoupro/code/image_captioning/notebook/finaldata.txt'
img_dir = '/home/vitoupro/code/image_captioning/notebook/test16'
# Robust custom loader to handle inconsistent spacing
data = []
with open(annotations_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            img_path, caption = parts
            data.append((img_path, caption))
        else:
            print(f"Skipping malformed line: {line}")

all_images = pd.DataFrame(data, columns=['image', 'caption'])


# Split dataset
train_images, eval_images, train_captions, eval_captions = train_test_split(
    all_images['image'].tolist(), all_images['caption'].tolist(), test_size=0.2, random_state=42
)

train_dataset = ImageCaptionDataset(
    img_labels=pd.DataFrame({'image': train_images, 'caption': train_captions}),
    img_dir=img_dir,
    vocab=word2idx,
    transform=transform,
    max_length=75
)

eval_dataset = ImageCaptionDataset(
    img_labels=pd.DataFrame({'image': eval_images, 'caption': eval_captions}),
    img_dir=img_dir,
    vocab=word2idx,
    transform=transform,
    max_length=75
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
encoder = EncoderCNN(embed_size=512).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx), num_layers=1,dropout_prob=0.3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

def custom_transform(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Return as list of words
    return text.split()

    

def calculate_wer(gt, pred, epoch, file_path='metric.txt'):
    content_pred = ''
    content_ground_true = ''

    match_pred = re.search(r"^(.*?)<END>", pred)
    if match_pred:
        content_pred = match_pred.group(1)

    match_ground_true = re.search(r"<START>(.*?)<END>", gt)
    if match_ground_true:
        content_ground_true = match_ground_true.group(1)

    with open(file_path, 'a') as file:  # Open file in append mode
        file.write(f"Epoch {epoch}\n")
        file.write("===========================\n")
        file.write(f"pred: {content_pred}\n")
        file.write(f"true: {content_ground_true}\n")
        file.write("===========================\n")

    # Ensure non-empty
    content_ground_true = content_ground_true or ''
    content_pred = content_pred or ''
    wer_score = jiwer.wer(content_ground_true, content_pred)

    return wer_score


def calculate_cer(gt, pred):
    content_pred = ''
    content_ground_true = ''

    match_pred = re.search(r"^(.*?)<END>", pred)
    if match_pred:
        content_pred = match_pred.group(1)

    match_ground_true = re.search(r"<START>(.*?)<END>", gt)
    if match_ground_true:
        content_ground_true = match_ground_true.group(1)

    return jiwer.cer(content_ground_true or '', content_pred or '')


def evaluate_model(encoder, decoder, dataloader, device, epoch):
    encoder.eval()
    decoder.eval()
    total_wer, total_cer, num_samples = 0, 0, 0
    skipped_samples = 0
    

    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            predicted_captions = outputs.argmax(-1)

            for i in range(len(captions)):
                gt_token_ids = captions[i].tolist()
                pred_token_ids = predicted_captions[i].tolist()

                gt_caption, gt_err = decode_indices(gt_token_ids, idx2word)
                pred_caption, pred_err = decode_indices(pred_token_ids, idx2word)
                
        


                if not gt_caption or not pred_caption:
                    print(f"\n[⚠️ Skipped Sample] Batch {batch_idx}, Index {i}, Epoch {epoch}")
                    print(f"  🔢 GT token IDs     : {gt_token_ids}")
                    print(f"  🔢 Pred token IDs   : {pred_token_ids}")
                    print(f"  ❌ GT decode error  : {gt_err}")
                    print(f"  ❌ Pred decode error: {pred_err}")
                    skipped_samples += 1
                    continue

                try:
                    wer = calculate_wer(gt_caption, pred_caption, epoch)
                    cer = calculate_cer(gt_caption, pred_caption)
                    
    
                except Exception as e:
                    print(f"\n[‼️ WER/CER Error] Sample {i} failed:")
                    print(f"  GT Caption : {gt_caption}")
                    print(f"  Pred Caption : {pred_caption}")
                    print(f"  Error : {e}")
                    skipped_samples += 1
                    continue

                total_wer += wer
                total_cer += cer
                num_samples += 1

    if num_samples == 0:
        print("⚠️ No valid samples evaluated.")
        return 1.0, 1.0

    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples
    
    print(f"\n✅ Epoch {epoch} Evaluation Summary:")
    print(f"   Avg WER  : {avg_wer:.3f}")
    print(f"   Avg CER  : {avg_cer:.3f}")
    print(f"   Skipped  : {skipped_samples} samples")
    
    return avg_wer, avg_cer



# Training Loop
num_epochs = 30
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
    wer, cer = evaluate_model(encoder, decoder, eval_loader, device, epoch)

    

    if wer < best_wer:
        best_wer = wer
        # Save the encoder and decoder models
        torch.save(encoder.state_dict(), 'encoder_finalv1test16.pth')
        torch.save(decoder.state_dict(), 'decoder_finalv1test16.pth')
        