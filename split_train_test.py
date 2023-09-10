import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math



class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, num_nodes=21):
        super(TransformerModel, self).__init__()

        # Generate position encoding
        position = torch.arange(num_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.position_enc = torch.zeros(num_nodes, d_model)
        self.position_enc[:, 0::2] = torch.sin(position * div_term)
        self.position_enc[:, 1::2] = torch.cos(position * div_term)

        # Add learnable group embedding
        self.embedding = nn.Linear(2, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 21, 2)  # x shape becomes [batch_size, 21, 2]

        # Embedding the input
        x = self.embedding(x)  # x shape becomes [batch_size, 21, d_model]

        # Add position encoding
        x += self.position_enc

        x = x.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model] as expected by Transformer
        x = self.encoder(x)
        x = self.classifier(x[0])  # Use the first token for classification
        return x
    def load_model(self,path):
        return torch.load(path)
# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 14)  # 假设有10个类别
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def load_model(self,path):
        return torch.load(path)
