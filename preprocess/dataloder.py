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
    from scipy import signal
    from scipy.fftpack import fft, ifft, fftfreq
    import matplotlib.pyplot as plt

    dataset = RailwaySensorDataset()

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for item in dataloader:
        print(item['x'].shape, item['x'][0, :, 0].shape)

        # plt.plot(item['x'][0, :, 0])
        # plt.show()
        plt.figure(figsize=(12, 6))

        data = item['x'][0, :, 0]

        plt.subplot(3, 1, 1)
        plt.plot(data)

        plt.subplot(3, 1, 2)
        F = fft(data.numpy())
        f = fftfreq(data.shape[0], 10)
        mask = np.where(f>0)
        plt.plot(f[mask], np.abs(F[mask]))
        # plt.show()

        plt.subplot(3, 1, 3)
        f, t, Sxx = signal.spectrogram(data, 10, nperseg=32)
        print(f.shape, t.shape, Sxx.shape)
        plt.pcolormesh(t, f, Sxx)
        plt.colorbar()
        plt.show()

        break


