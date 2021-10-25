import torch
import torch.nn as nn
import numpy as np
import math

from torch.autograd import Variable
import torch.nn.functional as F


class MDNRNN(nn.Module):
    def __init__(
            self,
            z_dim,
            action_dim,
            hidden_units=256,
            num_layers=1,
            n_gaussians=3,
            hidden_dim=256,
            device='cpu'):
        super(MDNRNN, self).__init__()

        self.n_gaussians = n_gaussians
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.hidden_units = hidden_units

        # Encoding
        self.fc1 = nn.Linear(self.z_dim + action_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, hidden_units, num_layers)
        self.lstm.flatten_parameters()

        # Output
        self.z_pi = nn.Linear(hidden_units, n_gaussians * self.z_dim)
        self.z_sigma = nn.Linear(hidden_units, n_gaussians * self.z_dim)
        self.z_mu = nn.Linear(hidden_units, n_gaussians * self.z_dim)

        self.device = torch.device(
            "cuda" if device == "cuda" else "cpu")

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_units,
            device=device)
        cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_units,
            device=device)
        return hidden, cell

    def reset(self, batch_size):
        self.hidden = self.init_hidden(batch_size, self.device)

    def forward(self, x, reset=False, encode=False):
        # x: (L, N, D)
        x = F.relu(self.fc1(x))
        # (L, N, D)
        if reset:
            self.hidden = self.init_hidden(x.shape[1], x.device)
            z, hidden = self.lstm(x, self.hidden)
        else:
            z, hidden = self.lstm(x, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())

        # z: (L, N, H), h_n: (num_layers, N, H), c_n: (num_layers, N, H)
        batch_size = x.size()[1]

        pi = self.z_pi(z).view(-1, batch_size, self.n_gaussians, self.z_dim)
        pi = F.softmax(pi, dim=2)
        sigma = torch.exp(self.z_sigma(z)).view(-1, batch_size,
                                                self.n_gaussians, self.z_dim)
        mu = self.z_mu(z).view(-1, batch_size, self.n_gaussians, self.z_dim)
        if encode:
            return z
        return pi, sigma, mu
