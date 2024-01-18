import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')

from preprocess.parse_data import get_data

class RailwaySensorDataset(Dataset):
    def __init__(self, is_train=True):
        root = '../data'
        dis_list = []
        fn_list = os.listdir(root+'/spectrogram')
        for fn in fn_list:
            wheel_id, dis, level, dir = fn[:-4].split('-')
            # print(fn, wheel_id, dis, level, dir)
            dis_int = int(dis)

            if dis_int not in dis_list:
                dis_list.append(dis_int)

        dis_list.sort()

        dis_train, dis_test = train_test_split(dis_list, train_size=0.7, random_state=0)
        dis_train.sort()
        dis_test.sort()

        if is_train:
            self.dis_list = dis_train
        else:
            self.dis_list = dis_test

        print('using dis: ', self.dis_list)

        # 里程（km）, 一位轮对磨耗（mm）, 二位轮对磨耗（mm）, 一位轮等效锥度, 二位轮等效锥度
        lma = pd.read_csv(f'../data/raw/others/LMA.csv', encoding='utf-8', usecols=(0, 1, 2, 3, 4))
        lma = np.array(lma)
        # print(lma.shape)
        lma_dict = {}
        for i in range(lma.shape[0]):
            # print(i)
            lma_dict[int(lma[i, 0])] = lma[i, 1:]
        self.lma_dict = lma_dict
        # print(self.lma_dict)

        data_list = []
        data_fft_list = []
        gt_list = []
        for fn in fn_list:
            wheel_id, dis, level, dir = fn[:-4].split('-')
            if int(dis) in self.dis_list:
                # print(fn)
                data_y = np.load(f'{root}/spectrogram/{wheel_id}-{dis}-{level}-Y.npy')
                data_z = np.load(f'{root}/spectrogram/{wheel_id}-{dis}-{level}-Z.npy')
                data = np.array([data_y, data_z])

                data_fft_y = np.load(f'{root}/fft/{wheel_id}-{dis}-{level}-Y.npy')
                data_fft_z = np.load(f'{root}/fft/{wheel_id}-{dis}-{level}-Z.npy')
                data_fft = np.array([data_fft_y, data_fft_z])

                # print(fn, data.shape)
                data_list.append(data)
                data_fft_list.append(data_fft)

                if int(wheel_id)<4:
                    gt_list.append(lma_dict[int(dis)][[0, 2]])
                else:
                    gt_list.append(lma_dict[int(dis)][[1, 3]])

                # gt_list.append()
        self.data_list = data_list
        self.gt_list = gt_list
        self.data_fft_list = data_fft_list

        self.data_len = len(self.data_list)
    def __len__(self):
        # print( 'sample index: ', len(self.data_list), len(self.gt_list) )
        # return len(self.dis_list)
        return self.data_len


    def __getitem__(self, idx):
        # print(idx)
        return {"spect": self.data_list[idx],
                "fft": self.data_fft_list[idx],
                "gt": self.gt_list[idx]}


if __name__=="__main__":
    from scipy import signal
    from scipy.fftpack import fft, ifft, fftfreq
    import matplotlib.pyplot as plt

    dataset = RailwaySensorDataset()
    dataset_test = RailwaySensorDataset(is_train=False)
    dataloader = DataLoader(dataset_test, batch_size=4)

    dataset_test.__getitem__(100)

    for item in dataloader:
        # print(item['data'].shape, item['gt'].shape)
        print(item['spect'].shape, item['fft'].shape, item['gt'].shape)

