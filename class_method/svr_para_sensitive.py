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



is_taper = 1
fft_train, y_train = load_data(trainDataLoader, is_taper)
fft_test, y_test = load_data(testDataLoader, is_taper)


# c_list = np.arange(0.02, 0.2, 0.005)
# mse_list = []
#
# for c in c_list:
#     svr = SVR(kernel='poly', C=c)
#     svr.fit(fft_train, y_train)
#
#     pred_train = svr.predict(fft_train)
#     pred_test = svr.predict(fft_test)
#
#     mse_i = mean_squared_error(pred_test, y_test)
#     # print(mse_i)
#     mse_list.append(mse_i)
#
# plt.plot(c_list, mse_list)
# plt.show()


epsilon_list = np.arange(0.001, 0.2, 0.005)
mse_list = []
cc_list = []
for epsilon in epsilon_list:
    svr = SVR(kernel='poly', epsilon=epsilon)
    svr.fit(fft_train, y_train)

    pred_train = svr.predict(fft_train)
    pred_test = svr.predict(fft_test)

    mse_i = mean_squared_error(pred_train, y_train)
    cc_i = pearsonr(pred_train, y_train)

    mse_list.append(mse_i)
    cc_list.append(cc_i[0])

# plt.figure(figsize=(10, 8))
fig, ax = plt.subplots()

ax.plot(epsilon_list, mse_list, c='g', lw=3, label='Mean Square Error')

ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Mean Square Error', fontsize=12, color='g')
ax.tick_params(axis ='y', labelcolor = 'g')

ax2 = ax.twinx()

ax2.plot(epsilon_list, cc_list, c='b', lw=3, label='Correlation Coefficient')
ax2.set_ylabel('Correlation Coefficient', fontsize=12, color='b')
ax2.tick_params(axis ='y', labelcolor = 'b')

plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.95)
# plt.legend()
# plt.show()


plt.savefig('para_sensitive.png', dpi=300)





