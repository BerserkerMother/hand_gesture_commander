import torch
from torch.utils import data

import pandas as pd


class HandGesture(data.Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        da = df.to_numpy()
        self.data = da[:, :-1]
        self.target = da[:, -1]

    def __getitem__(self, item):
        features = torch.tensor(self.data[item], dtype=torch.float)
        target = int(self.target[item])

        return features, target

    def __len__(self):
        return len(self.data)


HandGesture("data.csv")
