import torch
from torch.utils import data
import pandas as pd
import numpy as np


class Dataset(data.Dataset):

    def __init__(self, df, n_output, device):
        self.data = df
        self.device = device
        self.n_output = n_output

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.data.iloc[index].values, dtype=torch.float)
        y_indices = np.random.choice(self.data.index[self.data.index != index], size=self.n_output)
        y = torch.cat([torch.tensor(self.data.iloc[i].values, device=self.device, dtype=torch.float) for i in y_indices], dim=0)
        return x, y
