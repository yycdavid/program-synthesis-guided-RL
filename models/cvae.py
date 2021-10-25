import torch
import torch.nn as nn
import numpy as np
import math

from torch.autograd import Variable
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, d_map_in, d_flat, z_dim, side_len):
        super(ConvVAE, self).__init__()
        # Encoder
        # Input: (N, d_map_in, side_len, side_len)
        self.conv1 = nn.Conv2d(d_map_in, 32, 3, stride=1, padding=1)
        # (N, 32, side_len, side_len)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        # (N, 64, side_len/2, side_len/2)

        # Latent representation of mean and std
        half_side_len = math.floor(side_len / 2)
        self.after_conv_dim = 64 * half_side_len * half_side_len
        self.d_flat = d_flat
        pre_fc_dim = self.after_conv_dim + d_flat
        self.fc1 = nn.Linear(pre_fc_dim, z_dim)
        self.fc2 = nn.Linear(pre_fc_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, pre_fc_dim)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            64 *
            half_side_len *
            half_side_len,
            32,
            5,
            stride=2)  # (N, 32, 5, 5)
        self.deconv2 = nn.ConvTranspose2d(
            32,
            d_map_in,
            side_len - 5 + 1,
            stride=1)  # (N, d_map_in, side_len, side_len)

    def encode(self, x, flat_input):
        # x: (N, d_map_in, 10, 10), flat_input: (N, d_flat)
        h = F.relu(self.conv1(x))
        # (N, 32, 10, 10)
        h = F.relu(self.conv2(h))
        # (N, 64, 5, 5)
        h = h.view(-1, self.after_conv_dim)
        # (N, 64 * 5 * 5)
        h = torch.cat([h, flat_input], dim=1)
        # (N, 64 * 5 * 5 + d_flat)

        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        # z: (N, z_dim)
        h = self.fc3(z)
        # (N, 64 * 5 * 5 + d_flat)
        flat_decoded = h[:, self.after_conv_dim:]  # (N, d_flat)
        # (N, 64 * 5 * 5, 1, 1)
        h = h[:, :self.after_conv_dim].view(-1, self.after_conv_dim, 1, 1)
        h = F.relu(self.deconv1(h))
        # (N, 32, 5, 5)
        h = torch.sigmoid(self.deconv2(h))
        # (N, d_map_in, 10, 10)
        return h, torch.sigmoid(flat_decoded)

    def forward(self, x, flat_input, encode=False, mean=True):
        mu, logvar = self.encode(x, flat_input)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar


class CVAE(nn.Module):
    def __init__(
            self,
            dinp=784,
            dhid=400,
            drep=20,
            dout=20,
            dcond=0,
            cuda=False):
        # dhid = dimension of hidden layer
        # drep = dimension of hidden representation
        super(CVAE, self).__init__()

        self.dinp = dinp
        self.cuda = cuda
        self.fc1 = nn.Linear(dinp + dcond, dhid)
        self.fc21 = nn.Linear(dhid, drep)
        self.fc22 = nn.Linear(dhid, drep)
        self.fc3 = nn.Linear(drep + dcond, dhid)
        self.fc4 = nn.Linear(dhid, dout)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drep = drep

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, cond):
        size = x.size()
        inp = torch.cat((x.view(size[0], -1), cond), 1)
        mu, logvar = self.encode(inp)
        z = self.reparametrize(mu, logvar)
        z = torch.cat((z, cond), 1)
        return self.decode(z), mu, logvar
