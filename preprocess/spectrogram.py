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

            f, t, Sxx = signal.spectrogram(data_i, 10, nperseg=32)

            # print(f)
            # print(t)
            # print(Sxx)

            np.save(f'../data/spectrogram/{i}-{k}.npy', Sxx)


            # plt.subplot(4, 1, 1)
            # plt.plot(data_i)
            #
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
            # plt.subplot(4, 1, 4)
            # plt.pcolormesh(t, f, Sxx)
            # plt.colorbar()
            # plt.show()



