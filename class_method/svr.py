from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

import torch
import sys
sys.path.append('..')

from preprocess.dataloder import RailwaySensorDataset

# prepare the data
train_dataset = RailwaySensorDataset(is_train=True)
test_dataset = RailwaySensorDataset(is_train=False)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                              shuffle=True, num_workers=10, drop_last=False)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                             shuffle=False, num_workers=10)

fft = []
y = []
for item in trainDataLoader:
    fft_i, y_i = item['fft'], item['gt']
    print(fft_i.shape, y_i.shape)

    fft.append(fft_i)
    y.append(y_i)

fft = np.concatenate(fft).reshape((-1, 2000))
y = np.concatenate(y)[:, 1]

print(fft.shape, y.shape, type(fft))

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(fft, y)



fft = []
y = []
for item in trainDataLoader:
    fft_i, y_i = item['fft'], item['gt']
    print(fft_i.shape, y_i.shape)

    fft.append(fft_i)
    y.append(y_i)

fft = np.concatenate(fft).reshape((-1, 2000))
y = np.concatenate(y)[:, 1]

pred = regr.predict(fft)
print(np.array([pred, y]).T)

# n_samples, n_features = 10, 5
# rng = np.random.RandomState(0)
# y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)
#
# print(X.shape, y.shape)
#

#

