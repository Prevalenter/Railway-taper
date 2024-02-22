from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

import torch
import sys
sys.path.append('..')

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from preprocess.dataloder import RailwaySensorDataset
import matplotlib.pyplot as plt

from matplotlib import rcParams
# plt.rcParams["font.family"] = "Times New Roman"


# prepare the data
train_dataset = RailwaySensorDataset(is_train=True)
test_dataset = RailwaySensorDataset(is_train=False)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                              shuffle=True, num_workers=10, drop_last=False)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                             shuffle=False, num_workers=10)

def load_data(DataLoader, is_taper):
    fft = []
    y = []
    for item in DataLoader:
        fft_i, y_i = item['fft'], item['gt']
        print(fft_i.shape, y_i.shape)

        fft.append(fft_i)
        y.append(y_i)

    fft = np.concatenate(fft).reshape((-1, 2000))
    y = np.concatenate(y)[:, is_taper]

    return fft, y



is_taper = 0
fft_train, y_train = load_data(trainDataLoader, is_taper)
fft_test, y_test = load_data(testDataLoader, is_taper)


# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
# for ker in ['linear', 'poly', 'rbf', 'sigmoid']:
for ker in ['poly']:
    svr = SVR(kernel=ker, epsilon=0.001)
    svr.fit(fft_train, y_train)


    pred_train = svr.predict(fft_train)
    pred_test = svr.predict(fft_test)

    # calculate the accuracy
    print(ker)

    print('train: ', mean_squared_error(pred_train, y_train), pearsonr(pred_train, y_train))
    print('test: ', mean_squared_error(pred_test, y_test), pearsonr(pred_test, y_test))

    if ker=="poly":
        plt.scatter(y_train, pred_train, marker='s', facecolors='none', edgecolors='k', label='Train', s=80)
        plt.scatter(y_test, pred_test, marker='o', facecolors='none', edgecolors='b', label='Test', s=80)

        plt.xlabel('Ground Truth', fontsize=16)
        plt.ylabel('Prediction', fontsize=16)
        plt.subplots_adjust(top=0.9, right=0.95)
        plt.legend()

        if is_taper:
            type_str = 'Taper'
            plt.xlim(0, 0.7)
            plt.ylim(0, 0.7)
        else:
            type_str = 'Wear'
            plt.xlim(0, 2)
            plt.ylim(0, 2)
        # plt.grid()
        # plt.show()

        plt.savefig(type_str+'.png', dpi=300)
