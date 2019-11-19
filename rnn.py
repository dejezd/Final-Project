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

# Import the Dataset from PData, and process it for use in the model again.
# Need to create lists of entries from each channel of data, I think.

# Create the model
model = tf.keras.Sequential([
    #Needs some sort of embedding layer here, look into this more
    tf.keras.layers.Bidirectional(tf.kerasa.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
# Use the model on each of the different channels here (Maybe initialize it a bunch of times?)
