import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')

from parse_data import get_data

class RailwaySensorDataset(Dataset):
    def __init__(self):
        data_input, data_output = get_data()
        self.data_input = data_input
        self.data_output = data_output
        self.data_output[0] = [0, 0, 0, 0]
        self.sample_list = list(self.data_input.keys())

        print(self.data_output)
        print('-'*50)
    def __len__(self):
        return len(self.data_input.keys())

    def __getitem__(self, idx):
        key = self.sample_list[idx]
        wheel_sum, wheel1, wheel2 = self.data_input[key]
        print(key.split('-')[0])
        x = np.concatenate([wheel1, wheel2], axis=1)[:2000]
        y = np.array(self.data_output[int(key.split('-')[0])])

        # return image, label

        return {'x': x, 'data_out':y}


if __name__=="__main__":
    dataset = RailwaySensorDataset()

    dataloader = DataLoader(dataset, batch_size=2)

    for item in dataloader:
        print(item)



