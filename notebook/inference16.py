import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np

# Function to load idx2word and convert it to word2idx
def load_vocabulary(path):
    with open(path, 'r') as file:
        idx2word = json.load(file)
    word2idx = {v: int(k) for k, v in idx2word.items()}
    return idx2word, word2idx

# Load vocabulary
idx2word_path = '/home/vitoupro/code/image_captioning/notebook/idx2word.json'
idx2word, word2idx = load_vocabulary(idx2word_path)

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

    def forward(self, features, captions,states=None):
        embeddings = self.embed(captions)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        if states is None:
            states = (h0, c0)
    
        lstm_out, states = self.lstm(embeddings, states)
        outputs = self.linear(lstm_out)
        return outputs, states
        # lstm_out, _ = self.lstm(embeddings, (h0, c0))
        # outputs = self.linear(lstm_out)
        # return outputs

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
encoder = EncoderCNN(embed_size=512).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx), num_layers=1).to(device)

# Load model weights
encoder.load_state_dict(torch.load('encoder_finalv1.pth'))
decoder.load_state_dict(torch.load('decoder_finalv1.pth'))
encoder.eval()
decoder.eval()

# Beam Search Decoder
def beam_search_decoder(predictions, k):
    sequences = [[list(), 1.0]]  # list of (sequence, score)
    for row in predictions:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

# Function to generate a caption
def generate_caption(image_path, encoder, decoder, idx2word, word2idx, transform, beam_width=3):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        input_seq = torch.tensor([word2idx['<START>']]).to(device)
        sequences = [(input_seq, 0.0, None)]  # (sequence, log_prob, states)

        for _ in range(20):  # Max length of the caption
            all_candidates = []
            for seq, score, states in sequences:
                if seq[-1] == word2idx['<END>']:
                    all_candidates.append((seq, score, states))
                    continue
                outputs, states = decoder(features,seq.unsqueeze(0), states)
                softmaxed = torch.softmax(outputs[:, -1], dim=1)
                top_indices = torch.topk(softmaxed, beam_width)[1].squeeze(0)
                
                for idx in top_indices:
                    next_seq = torch.cat([seq, idx.unsqueeze(0)])
                    next_score = score + torch.log(softmaxed[0, idx]).item()
                    all_candidates.append((next_seq, next_score, states))

            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

        best_seq, best_score, _ = sequences[0]
        caption = ' '.join([idx2word[str(idx.item())] for idx in best_seq if idx.item() not in [word2idx['<START>'], word2idx['<END>']]])
        return caption

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
])

# Image path
image_path = '/home/vitoupro/code/image_captioning/data/image.png'

caption = generate_caption(image_path, encoder, decoder, idx2word, word2idx, transform)
print("Generated Caption:", caption.replace(" ", ""))
