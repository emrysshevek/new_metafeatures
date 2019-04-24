import torch
from torch import nn
from torch import optim

import pandas as pd
import numpy as np


class Encoder:

    def __init__(self, input_dim=50, n_output=10, embed_dim=10, epoch_length=None, n_epochs=50, lr=0.001, loss=nn.MSELoss()):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = n_output * input_dim
        self.embed_dim = embed_dim
        self.epoch_length = epoch_length
        self.n_epochs = n_epochs
        self.lr = lr
        self.loss = loss

    def encode(self, generator):
        encoder = nn.ModuleList([
            nn.Linear(self.input_dim, self.embed_dim),
            # nn.ReLU(),
            nn.Linear(self.embed_dim, self.output_dim),
            # nn.ReLU()
        ])

        self._train_encoder(nn.Sequential(*encoder), generator)
        encoding = torch.zeros(len(generator.dataset), self.embed_dim, requires_grad=False)
        for i, (x, _) in enumerate(generator):
            encoding[i] = encoder[0](x)
        return encoding.mean(dim=0).detach()

    def _train_encoder(self, encoder, generator):
        opt = optim.Adam(encoder.parameters(), lr=self.lr)
        for i in range(self.n_epochs):
            self._run_epoch(encoder, generator, opt)

    def _run_epoch(self, encoder, generator, opt):
        for x, y in generator:
            pred = encoder(x)
            loss = self.loss(pred, y)
            loss.backward()
            opt.zero_grad()
            opt.step()

