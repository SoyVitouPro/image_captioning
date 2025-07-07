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
idx2word_path = '/home/vitoupro/code/image_captioning/notebook/idx2word_finalword_khmercut.json'
idx2word, word2idx = load_vocabulary(idx2word_path)

# Encoding a list of words (word-level)
def encode_khmer_sentence(sentence, word2idx):
    words = sentence.strip().split()
    indices = []
    for word in words:
        index = word2idx.get(word)
        if index is None:
            return None, f"Word '{word}' not found in vocabulary!"
        indices.append(index)
    return indices, None

# Decoding a list of indices (word-level)
def decode_indices(indices, idx2word):
    words = []
    for index in indices:
        word = idx2word.get(str(index))
        if word is None:
            return None, f"Index '{index}' not found in idx2word!"
        words.append(word)
    return ' '.join(words), None

# Attention Module
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, hidden):
        hidden = hidden.unsqueeze(1).repeat(1, encoder_out.size(1), 1)
        attn_input = torch.cat((encoder_out, hidden), dim=2)
        energy = torch.tanh(self.attn(attn_input))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for name, param in resnet.named_parameters():
            param.requires_grad = 'layer4' in name
        modules = list(resnet.children())[:-2]  # Keep conv feature map
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, H, W)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)  # (B, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, 196, 2048)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, encoder_dim=2048, num_layers=1, dropout_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)
        vocab_size = self.linear.out_features

        # Init hidden state and cell state
        h = self.init_h(encoder_out.mean(dim=1))  # (B, hidden_size)
        c = self.init_c(encoder_out.mean(dim=1))

        outputs = torch.zeros(batch_size, seq_len - 1, vocab_size).to(encoder_out.device)

        for t in range(seq_len - 1):
            embeddings = self.dropout(self.embed(captions[:, t]))  # (B, embed_size)
            context, _ = self.attention(encoder_out, h)            # (B, encoder_dim)
            lstm_input = torch.cat([embeddings, context], dim=1)   # (B, embed + encoder_dim)
            h, c = self.lstm(lstm_input, (h, c))                   # LSTMCell
            output = self.linear(self.dropout(h))                 # (B, vocab_size)
            outputs[:, t, :] = output

        return outputs

# Image Captioning Dataset
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, img_labels, img_dir, vocab, transform=None, max_length=20):
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
        indices, error = encode_khmer_sentence(caption, self.vocab)
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
annotations_file = '/home/vitoupro/code/image_captioning/notebook/finaldataword_khmercut.txt'
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
    max_length=20
)

eval_dataset = ImageCaptionDataset(
    img_labels=pd.DataFrame({'image': eval_images, 'caption': eval_captions}),
    img_dir=img_dir,
    vocab=word2idx,
    transform=transform,
    max_length=20
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
encoder = EncoderCNN().to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx), num_layers=1,dropout_prob=0.3).to(device)

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
        for batch_idx, (images, captions) in enumerate(dataloader):
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

    return avg_wer, avg_cer, bleu1, bleu2,bleu3, bleu4



# Training Loop
num_epochs = 30
best_wer = float('inf')

train_losses, wer_scores, cer_scores = [], [], []
bleu1_scores, bleu2_scores,bleu3_scores, bleu4_scores = [], [], [],[]

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)

    # Slice input and target
        inputs = captions[:, :-1]      # All tokens except last
        targets = captions[:, 1:-1]     # All tokens except first

        outputs = decoder(features, inputs)  # [B, T-1, V]

    # Sanity check (optional)
        assert outputs.size(1) == targets.size(1), f"Mismatch: outputs {outputs.shape}, targets {targets.shape}"

    # Reshape for loss
        outputs = outputs.reshape(-1, outputs.size(-1))  # [B*(T-1), V]
        targets = targets.reshape(-1)                   # [B*(T-1)]

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader)}')
    wer, cer, bleu1, bleu2,bleu3, bleu4 = evaluate_model(encoder, decoder, eval_loader, device, epoch)

     # ✅ Append all metrics here
    train_losses.append(total_loss / len(train_loader))
    wer_scores.append(wer)
    cer_scores.append(cer)
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)
    
    
    if wer < best_wer:
        best_wer = wer
        # Save the encoder and decoder models
        torch.save(encoder.state_dict(), 'encoder_finalv1_att_word.pth')
        torch.save(decoder.state_dict(), 'decoder_finalv1_att_word.pth')
        


epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(16, 10))

# WER and CER
plt.subplot(2, 2, 1)
plt.plot(epochs, wer_scores, label='WER', marker='o')
plt.plot(epochs, cer_scores, label='CER', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('WER & CER Over Epochs')
plt.legend()
plt.grid(True)

# BLEU Scores
plt.subplot(2, 2, 2)
plt.plot(epochs, bleu1_scores, label='BLEU-1', marker='o')
plt.plot(epochs, bleu2_scores, label='BLEU-2', marker='x')
plt.plot(epochs, bleu3_scores, label='BLEU-3', marker='s')
plt.plot(epochs, bleu4_scores, label='BLEU-4', marker='^')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.title('BLEU Scores Over Epochs')
plt.legend()
plt.grid(True)

# Training Loss
plt.subplot(2, 2, 3)
plt.plot(epochs, train_losses, label='Train Loss', color='orange', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_evaluation_metrics_word.png", dpi=300)
plt.show()



        