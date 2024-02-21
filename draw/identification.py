import sys
from scipy import signal
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')

from preprocess.parse_data import get_data

if __name__=="__main__":
    data_input, data_output = get_data('../data/raw')

    print(list(data_input.keys()))

    # for k in data_input.keys():

    idx = 0
    plt.figure(figsize=(10, 10))
    for k in ['40000-HIGH-Y', '80000-HIGH-Y', '160000-HIGH-Y', '240000-HIGH-Y']:
    # for k in ['50000-LOW-Y', '90000-LOW-Y', '180000-LOW-Y', '250000-LOW-Y']:
    # for k in ['0-LOW-Z', '90000-LOW-Z', '180000-LOW-Z', '250000-LOW-Z']:
        data_wheel = data_input[k]
        print(k, data_wheel.shape)

        # for i in range(8):
        i = 0
        data_i = data_wheel[:, i]
        print(i, data_i.shape)

        f, t, Sxx = signal.spectrogram(data_i, 10, nperseg=32)

        print(f.shape, t.shape, Sxx.shape)

        plt.subplot(4, 2, 1+idx*2)
        plt.plot(data_i, c='k')
        plt.title(k)

        # plt.subplot(4, 1, 2)
        # F = fft(data_i)
        # freq = fftfreq(data_i.shape[0], 10)
        # mask = np.where(freq > 0)
        # plt.plot(freq[mask], np.abs(F[mask]))
        #
        # plt.subplot(4, 1, 3)
        #
        # plt.imshow(Sxx)
        # plt.colorbar()
        #
        plt.subplot(4, 2, 2+idx*2)
        plt.pcolormesh(t, f, Sxx)
        plt.colorbar()
        idx += 1
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3)
    # plt.show()

    plt.savefig('ident.png', dpi=400)



