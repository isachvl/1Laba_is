
import torch
from torch import nn

class DigitMLP(nn.Module):

    def __init__(self, n_layers, hidden, dropout): #dropout случайное отключение части нейронов
        super().__init__()

        layers = []
        input_size = 64

        for i in range(n_layers):
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            input_size = hidden

        # Выходной слой для 10 цифр
        layers.append(nn.Linear(input_size, 10))

        self.network = nn.Sequential(*layers) # кдает все слои в список обхъектов 

    def forward(self, x):
        return self.network(x)