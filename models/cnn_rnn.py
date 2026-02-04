# models/cnn_rnn.py

import torch
import torch.nn as nn

class CNNRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)  
        b, c, h, w = x.size()

        x = x.mean(dim=2)          # collapse height
        x = x.permute(0, 2, 1)     # (batch, width, channels)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x
