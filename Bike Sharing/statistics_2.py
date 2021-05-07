################################################################################
# statistics_2.py
# AUTHOR:           NGOC TRAN
# CREATED:          07 May 2021
# DESCRIPTION:      Survey statistics of shuffled streaming data.
################################################################################
import csv
import os
import matplotlib.pyplot as plt

local_path = os.getcwd()
SOURCE = 'Bike Sharing/datasets/hour_ram_shuffled.csv'
SEP = ','
running_mean = list()
running_std = list()

with open(local_path + '/' + SOURCE) as R:
    iter = csv.DictReader(R, delimiter=SEP)
    x = 0.0
    x_squared = 0.0
    for n, row in enumerate(iter):
        temp = float(row['temp'])
        if n == 0:
            max_x, min_x = temp, temp  # Initialize
        else:
            max_x, min_x = max(temp, max_x), min(temp, min_x)
        x += temp
        x_squared += temp ** 2
        running_mean.append(x / (n + 1))
    print('Feature \'temp \': mean=%0.3f, max=%0.3f, min=%0.3f' % (running_mean[-1],
                                                                             max_x, min_x))

plt.plot(running_mean, 'r-', label='mean')
plt.xlabel('Number of training examples')
plt.ylabel('Value')
plt.show()