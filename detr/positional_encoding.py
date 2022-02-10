'''
borrow code from 
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
'''
import numpy as np
import seaborn as sns

d_hid = 128
n_position = 50


def get_position_angle_vec(position):
    return [
        position / np.power(10000, 2 * (hid_j // 2) / d_hid)
        for hid_j in range(d_hid)
    ]


sinusoid_table = np.array(
    [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

tables = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        tables[i][j] = sum(
            [a * b for a, b in zip(sinusoid_table[i], sinusoid_table[j])])

for i in range(50):
    for j in range(0, i):
        tables[i][j] = tables[j][i]

plot = sns.heatmap(tables, cmap="Blues")
fig = plot.get_figure()
fig.savefig('./test.png')
