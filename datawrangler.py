##
# Data Wrangler for EMG Classification RNN
# Jeremy Decker
# 11/14/19
# This Program will create a series of entries by gesture in each file in the data, to be further processed in the
# model itself.
##

import pandas as pd
import numpy as np
import os

print(os.listdir('data/'))

EMG1 = pd.read_csv("data/1_raw_data_13-12_22.03.16.txt", header=0, sep='\t')
print(EMG1.head())

# I need to write up a script that creates a new "small" table after the class changes and write the ones that are
# Non-zero into individual text files. 