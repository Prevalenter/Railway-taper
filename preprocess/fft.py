import sys
from scipy import signal
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from parse_data import get_data

if __name__=="__main__":
    data_input, data_output = get_data('../data/raw')

    print(list(data_input.keys()))

    for k in data_input.keys():
        data_wheel = data_input[k]
        print(k, data_wheel.shape)

        for i in range(8):
            data_i = data_wheel[:, i]
            print(i, data_i.shape)

            plt.subplot(2, 1, 1)
            plt.plot(data_i)

            plt.subplot(2, 1, 2)
            F = fft(data_i)
            freq = fftfreq(data_i.shape[0], 10)
            mask = np.where(freq > 0)
            # plt.plot(freq[mask], np.abs(F[mask]))
            # plt.show()
            # print(freq)
            print(freq[mask].shape, np.abs(F[mask]).shape)

            np.save(f'../data/fft/{i}-{k}.npy', np.abs(F[mask]))
