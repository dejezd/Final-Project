##
# Recurrent Neural Network for Gesture Classification from 8-channel EMG data
# Jeremy Decker
# 11/19/19
# This program run each channel of EMG data through a separate LSTM and use a simple majority to classify the data
# May eventually use another NN to further process this output.
##

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


