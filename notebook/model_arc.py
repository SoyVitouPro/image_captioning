import torch.nn as nn
from torchvision import models
import torch

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]  # Remove the last layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)  # Match LSTM hidden size

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # Ensure features match the hidden size of LSTM
        features = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        h0 = features  # LSTM hidden state
        c0 = torch.zeros_like(h0)  # LSTM cell state
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(outputs)
        return outputs
