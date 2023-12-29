import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_csv = pd.read_csv('../data/LMA.csv', encoding='utf-8',usecols=(0,1,2,3,4))

data = np.array(data_csv)
# print(data)

for i in range(4):
	plt.subplot(2, 2, 1+i)
	plt.scatter(data[:, 0], data[:, 1+i])
	# plt.xlabel('')
	plt.xlabel('Mileage')
	plt.ylabel('Equivalent taper')

plt.subplots_adjust(left=0.1, right=0.95)

plt.show()





