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
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu, SmoothingFunction

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
        for name, param in resnet.named_parameters():
            param.requires_grad = 'layer4' in name
        modules = list(resnet.children())[:-2]  # Keep conv feature map
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.embed = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, H, W)
        features = self.adaptive_pool(features)  # (B, 2048, 14, 14)
        features = features.mean(dim=[2, 3])  # (B, 2048) - Global average pooling
        features = self.embed(features)  # (B, embed_size)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(512, hidden_size)  # Initialize from encoder output (512)
        self.init_c = nn.Linear(512, hidden_size)  # Initialize from encoder output (512)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        lstm_out = self.dropout(lstm_out)
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
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
]) 

# Load dataset
annotations_file = '/home/vitoupro/code/image_captioning/notebook/finaldata.txt'
img_dir = '/home/vitoupro/code/image_captioning/notebook/downloaded_images'
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models  
encoder = EncoderCNN(embed_size=512).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx), num_layers=1, dropout_prob=0.3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
params = list(decoder.parameters()) + list(encoder.parameters())
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

    

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re
import jiwer

smoothing = SmoothingFunction().method4  # More stable smoothing

def calculate_wer(gt, pred, epoch, file_path='metric.txt'):
    match_pred = re.search(r"^(.*?)<END>", pred)
    content_pred = match_pred.group(1) if match_pred else ""

    match_gt = re.search(r"<START>(.*?)<END>", gt)
    content_gt = match_gt.group(1) if match_gt else ""

    with open(file_path, 'a') as file:
        file.write(f"Epoch {epoch}\n")
        file.write("===========================\n")
        file.write(f"pred: {content_pred}\n")
        file.write(f"true: {content_gt}\n")
        file.write("===========================\n")

    return jiwer.wer(content_gt or '', content_pred or '')

def calculate_cer(gt, pred):
    match_pred = re.search(r"^(.*?)<END>", pred)
    content_pred = match_pred.group(1) if match_pred else ""

    match_gt = re.search(r"<START>(.*?)<END>", gt)
    content_gt = match_gt.group(1) if match_gt else ""

    return jiwer.cer(content_gt or '', content_pred or '')

def evaluate_model(encoder, decoder, dataloader, device, epoch):
    encoder.eval()
    decoder.eval()

    total_wer, total_cer = 0, 0
    num_samples = 0
    skipped = 0

    all_references = []
    all_predictions = []

    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])  # Predict T-1 tokens
            predicted_captions = outputs.argmax(-1)

            for i in range(captions.size(0)):
                gt_ids = captions[i].tolist()
                pred_ids = predicted_captions[i].tolist()

                gt_text, gt_err = decode_indices(gt_ids, idx2word)
                pred_text, pred_err = decode_indices(pred_ids, idx2word)

                if not gt_text or not pred_text or gt_err or pred_err:
                    skipped += 1
                    continue

                # Extract only content between <START> and <END>
                ref_match = re.search(r"<START>(.*?)<END>", gt_text)
                pred_match = re.search(r"^(.*?)<END>", pred_text)
                reference = ref_match.group(1).strip() if ref_match else ""
                prediction = pred_match.group(1).strip() if pred_match else ""

                if not reference or not prediction:
                    continue

                wer = calculate_wer(gt_text, pred_text, epoch)
                cer = calculate_cer(gt_text, pred_text)
                total_wer += wer
                total_cer += cer
                num_samples += 1

                # Use character-level tokens for BLEU
                ref_chars = list(reference)
                pred_chars = list(prediction)

                if len(ref_chars) == 0 or len(pred_chars) == 0:
                    continue

                all_references.append([ref_chars])  # list of one reference
                all_predictions.append(pred_chars)

    # Corpus-level BLEU
    bleu1 = corpus_bleu(all_references, all_predictions, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(all_references, all_predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(all_references, all_predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(all_references, all_predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    avg_wer = total_wer / num_samples if num_samples else 1.0
    avg_cer = total_cer / num_samples if num_samples else 1.0

    print(f"\n✅ Epoch {epoch} Evaluation Summary:")
    print(f"   WER     : {avg_wer:.3f}")
    print(f"   CER     : {avg_cer:.3f}")
    print(f"   BLEU-1  : {bleu1:.3f}")
    print(f"   BLEU-2  : {bleu2:.3f}")
    print(f"   BLEU-3  : {bleu3:.3f}")
    print(f"   BLEU-4  : {bleu4:.3f}")
    print(f"   Skipped : {skipped} samples")

    return avg_wer, avg_cer, bleu1, bleu2, bleu3, bleu4



# Training Loop
num_epochs = 50
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# Scheduled Sampling
scheduled_sampling_start = 0.0
scheduled_sampling_end = 0.3
scheduled_sampling_increase = (scheduled_sampling_end - scheduled_sampling_start) / num_epochs

# Label Smoothing (Optional)
def label_smoothing_loss(pred, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(pred, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    return (confidence * nll_loss + smoothing * smooth_loss).mean()

# Scheduler based on BLEU-4 improvement
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Early stopping variables
best_bleu1 = 0.0
early_stopping_patience = 5
epochs_no_improve = 0

train_losses, wer_scores, cer_scores = [], [], []
bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    sampling_prob = min(scheduled_sampling_start + epoch * scheduled_sampling_increase, scheduled_sampling_end)
    
    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        
        # Apply scheduled sampling during training
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        
        batch_size, seq_len = inputs.size()
        outputs = torch.zeros(batch_size, seq_len, len(word2idx)).to(device)
        
        for t in range(seq_len):
            if t == 0 or torch.rand(1).item() >= sampling_prob:
                current_input = inputs[:, t:t+1]  # Use ground truth
            else:
                current_input = outputs[:, t-1].argmax(dim=-1).unsqueeze(1)  # Use previous prediction
            
            output = decoder(features, current_input)
            outputs[:, t] = output.squeeze(1)
        
        loss = criterion(outputs.view(-1, len(word2idx)), targets.reshape(-1))
        # Alternative: enable label smoothing
        # loss = label_smoothing_loss(outputs.view(-1, len(word2idx)), targets.reshape(-1), smoothing=0.1)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}')
    wer, cer, bleu1, bleu2, bleu3, bleu4 = evaluate_model(encoder, decoder, eval_loader, device, epoch)
    scheduler.step(bleu4)
    
    # Store metrics for plotting
    train_losses.append(total_loss / len(train_loader))
    wer_scores.append(wer)
    cer_scores.append(cer)
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)
    
    # Early stopping based on BLEU-4
    if bleu1 > best_bleu1:
        best_bleu1 = bleu1
        epochs_no_improve = 0
        torch.save(encoder.state_dict(), 'encoder_experiment16_best.pth')
        torch.save(decoder.state_dict(), 'decoder_experiment16_best.pth')
        print(f"🎯 New best BLEU-1: {best_bleu1:.4f} - Model saved!")
    else:
        epochs_no_improve += 1
        print(f"🔁 No BLEU-1 improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= early_stopping_patience:
        print(f"⏹️ Early stopping triggered at epoch {epoch+1} (no BLEU-1 improvement in {early_stopping_patience} epochs).")
        break

# Plotting training metrics
epochs_completed = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(16, 10))

# WER and CER
plt.subplot(2, 2, 1)
plt.plot(epochs_completed, wer_scores, label='WER', marker='o')
plt.plot(epochs_completed, cer_scores, label='CER', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('WER & CER Over Epochs')
plt.legend()
plt.grid(True)

# BLEU Scores
plt.subplot(2, 2, 2)
plt.plot(epochs_completed, bleu1_scores, label='BLEU-1', marker='o')
plt.plot(epochs_completed, bleu2_scores, label='BLEU-2', marker='x')
plt.plot(epochs_completed, bleu3_scores, label='BLEU-3', marker='s')
plt.plot(epochs_completed, bleu4_scores, label='BLEU-4', marker='^')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.title('BLEU Scores Over Epochs')
plt.legend()
plt.grid(True)

# Training Loss
plt.subplot(2, 2, 3)
plt.plot(epochs_completed, train_losses, label='Train Loss', color='orange', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_evaluation_metrics_experiment16.png", dpi=300)
plt.show()
        