import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def get_data(root='../data/'):
	fn_list = os.listdir(root)
	# print(fn_list)
	usecols = (1, 2, 3, 4,
	           5, 6, 7, 8)

	data_input = {}
	for fn in fn_list:
		if fn in ['others', '.DS_Store']:
			continue
		# print(fn)
		data_csv = pd.read_csv(f'{root}/{fn}', encoding='gb18030', usecols=usecols)
		data = np.array(data_csv)

		data_wheel = data[:, [0, 1, 6, 7, 2, 3, 4, 5]]
		# data_wheel2 = data[:, []]

		dis, level, dir = fn[:-4].split('-')
		dis = int(dis)

		# print(data.shape, data_wheel1.shape, data_wheel2.shape)
		# print(dis, level, dir, fn[:-4])

		data_input[fn[:-4]] = data_wheel

	data_lma = pd.read_csv(f'{root}/others/LMA.csv', encoding='utf-8',usecols=(0,1,2,3,4))
	data_lma = np.array(data_lma)

	# print(data_lma.shape)
	# print(data_lma)

	rst = {}
	for data_i in (data_lma):
		# print(data_i)
		rst[int(data_i[0])] = list(data_i[1:])

	# print(rst)

	return data_input, rst
