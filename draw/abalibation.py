import numpy as np
import matplotlib.pyplot as plt

mode_size = {
    'tiny224': 0.0047476,
    'small224': 0.004855,
    'base224': 0.004036,
    'base384': 0.004943
}

# plt.figure(figsize=(8, 3))
# print(list(mode_size.values()))
# plt.bar( list(mode_size.keys()), list(mode_size.values()) , ec='k', width=0.4 )
# plt.xlabel('Model size', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.ylim(0.0038, 0.005)
# plt.subplots_adjust(left=0.15, bottom=0.2)
# # plt.show()
# plt.savefig('abla_mode_size.png', dpi=400)

stride = {
    '3': 0.004577, '5': 0.004199, '7': 0.004468,
    '9': 0.005250, '11': 0.005121, '13': 0.005074,
}

# plt.figure(figsize=(8, 3))
# plt.plot( list(stride.keys()), list(stride.values()), '-o', c='k' )
# plt.grid()
# plt.xlabel('Stride', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.subplots_adjust(left=0.12, bottom=0.2)
# # plt.show()
# plt.savefig('abla_stride.png', dpi=400)

lr = {
    '0.01': 0.01458,
    '0.005': 0.00555,
    '0.001': 0.004199,
    '0.0005': 0.0041089,
    '0.0001': 0.0033194,
    '0.00005': 0.0035659,
    '0.00001': 0.0041930
}

#
lr_mohao = {
    '0.01': 0.2417573,
    '0.005': 0.05912,
    '0.001': 0.062232,
    '0.0005': 0.04684,
    '0.0001': 0.030617,
    '0.00005': 0.0283405,
    '0.00001': 0.0316502
}

fig, ax = plt.subplots()

ax.plot( list(lr_mohao.keys()), list(lr_mohao.values()), c='g', lw=2, label='MSE of mohao')

ax.set_xlabel('learning rate', fontsize=12)
ax.set_ylabel('MSE of mohao', fontsize=12, color='g')
ax.tick_params(axis ='y', labelcolor = 'g')

ax2 = ax.twinx()
ax2.plot( list(lr.keys()), list(lr.values()), c='b', lw=2, label='MSE of taper')
# ax2.plot(epsilon_list, cc_list, c='b', lw=3, label='Correlation Coefficient')
ax2.set_ylabel('MSE of taper', fontsize=12, color='b')
ax2.tick_params(axis ='y', labelcolor = 'b')

plt.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.95, )
# plt.legend()
# plt.show()
plt.savefig('abla_lr.png', dpi=300)