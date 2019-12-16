##
# Data Wrangler for EMG Classification RNN
# Jeremy Decker
# 11/14/19
# This program will create a tensorflow database containing arrays for each file in the dataset,
# and its appropriate label, for analysis in the rnntest.py file.
##
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def main():
    list_emg = tf.data.Dataset.list_files(str('PData/emg_sample_*.csv'))
    labeled_ds = list_emg.map(process_data)
def process_data(file_path):
    file = np.loadtxt(file_path, delimiter=',')
    data = file[0:8, :]  # Should input all 8 columns of data needed
    label = file[1, 8]
    return data, label
