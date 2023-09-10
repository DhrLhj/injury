#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
import torch.nn as nn
import math
# from flask import Flask

# app = Flask(__name__)

# @app.route('/main')
def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ws_demo.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

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
if __name__ == '__main__':
    main()
