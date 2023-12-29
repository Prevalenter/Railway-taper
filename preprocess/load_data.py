import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

root = '../data/'
fn_list = os.listdir(root)

# data = np.load()
# print(fn_list)

usecols = (1, 2, 3, 4,
           5, 6, 7, 8)

for fn in fn_list:
	if fn in ['others', '.DS_Store']:
		continue
	# , encoding = 'gb18030')
	# print(fn)
	data_csv = pd.read_csv(f'../data/{fn}', encoding='gb18030', usecols=usecols)
	data = np.array(data_csv)

	data_wheel1 = data[:, [0, 2, 4, 6]]
	data_wheel2 = data[:, [1, 3, 5, 7]]

	dis, level, dir = fn[:-4].split('-')
	dis = int(dis)


	# print()
	print(data.shape, data_wheel1.shape, data_wheel2.shape)
	print(dis, level, dir)


data_rst = pd.read_csv('../data/others/LMA.csv', encoding='utf-8',usecols=(0,1,2,3,4))
data_rst = np.array(data_rst)

print(data_rst.shape)
print(data_rst)

rst = {}
for data_i in (data_rst):
	print(data_i)
	rst[data_i[0]] = list(data_i[1:])

print(rst)
