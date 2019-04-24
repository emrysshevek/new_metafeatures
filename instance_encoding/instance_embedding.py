import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from data.dataset import Dataset
from instance_encoding.encoder import Encoder



df = pd.read_csv("/users/guest/m/masonfp/Desktop/new_metafeatures/data/accuracy_10min.csv")

input_dim = 50
n_output = 10
embed_dim = 10
batch_size = 16

encoder = Encoder(input_dim=input_dim, n_output=n_output, embed_dim=embed_dim)
for i, instance in df.iterrows():

    data = pd.read_csv(instance['data_path'], index_col=0)
    data = pd.get_dummies(data)
    data = (data - data.min()) / (data.max() - data.min())
    data = data.fillna(-1)
    data = pd.DataFrame(PCA().fit_transform(data))
    if data.shape[1] >= input_dim:
        data = data[data.columns[:input_dim]]
    else:
        data = pd.concat([data, pd.DataFrame(np.zeros((data.shape[0], input_dim-data.shape[1])))], axis=1)
    data = Dataset(data, n_output, None)
    generator = DataLoader(data, batch_size=batch_size, shuffle=True)

    enc = encoder.encode(generator)
    print(enc)
    break



