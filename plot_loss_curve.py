import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

def find_floats(string):
    # return re.findall(r'\b[-]?\d+(\.\d+)?\b', string)
    return re.findall(r"\d+\.\d+", string)

log_file = "./logs/1009_lyon2024_sgd.out"
save_scatter = log_file[:-4] + ".png"
total_epoch = 500
epoch_flag = '/' + str(total_epoch - 1)
print(epoch_flag)
# save_scatter = "./logs/1009_lyon2024_load1008pretrained.png"

f = open(log_file)
info = f.read().split('Total network parameters:')[1:]

info = info[0].split('\n')
# tverskyloss_buff, focalloss_buff, totalloss_buff = 0, 0, 0
TverskyLoss, FocalLoss, TotalLoss = [], [], []
epochs = []
for t_info in tqdm(info):
    if epoch_flag in t_info:
        epoch = re.findall(r"\d+", t_info[:t_info.find(epoch_flag)])
        epoch_buffer = int(float(epoch[0]))

        loss_info = find_floats(t_info)
        tverskyloss, focalloss, totalloss = float(loss_info[0]), float(loss_info[1]), float(loss_info[2])
    elif " 0/16" in t_info:
        tverskyloss_buff, focalloss_buff, totalloss_buff = tverskyloss, focalloss, totalloss
        TverskyLoss.append(tverskyloss_buff)
        FocalLoss.append(focalloss_buff)
        TotalLoss.append(totalloss_buff)

        epochs.append(epoch_buffer)

# 合并到一张曲线图上
fig = plt.figure(figsize=(20, 5))

plt.scatter(epochs, TverskyLoss, s=0.1, color='blue')
plt.plot(epochs, TverskyLoss, color = 'blue', label='TverskyLoss')

plt.scatter(epochs, FocalLoss, s=0.1, color='green')
plt.plot(epochs, FocalLoss, color = 'green', label='FocalLoss')

plt.scatter(epochs, TotalLoss, s=0.1, color='red')
plt.plot(epochs, TotalLoss, color = 'red', label='TotalLoss')

plt.legend()
plt.title(save_scatter[save_scatter.rfind('/')+1:-4], fontsize=14)
plt.show()
# plt.savefig(save_scatter)

