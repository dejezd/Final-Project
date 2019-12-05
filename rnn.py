##
# Recurrent Neural Network for Gesture Classification from 8-channel EMG data
# Jeremy Decker
# 11/19/19
# This program run each channel of EMG data through a separate LSTM and use a simple majority to classify the data
# May eventually use another NN to further process this output.
##

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os

# Import a list of file names to split up to form the dataset
files_path = 'PData/'
dataset = [None]*872
fnames = glob.glob(os.path.join(files_path, 'emg_sample_*.csv'))
i = 0
# Create a data structures to separate instances into numpy arrays to put into the model
for names in fnames:
    dataset[i] = np.loadtxt(names, delimiter=',')
# Load in labels vector
flabels = np.loadtxt('PData/emg_labels.csv', delimiter=',')
##
# Create a loop with sequentially named datasets to set up testing and traning filenames
# Need to parse it down using the data.shape file, and set the data to data[:, 1:9]
# Set Labels to add data[0, 10], to a list for use as labels.
#print('Creating Test and Training Data sets')
train, test, train_l, test_l = train_test_split(dataset, flabels, test_size=0.25, random_state=1)
#print(np.size(train))
#print(np.size(test))
##

# Start to set up the data for the sets
# Need to find a way to run the data for the model by reading the csv, and identifying the labels from it.
# Resource: https://datascience.stackexchange.com/questions/51249/training-keras-model-with-multiple-csv-files
# LSTM Start
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)), # Will play around with more hyperparameters here.
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
history = model.fit(train, epochs=10,
                    validation_data=test,
                    validation_steps=30)
test_loss, test_acc = model.evaluate(test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# Insert Plotting functions here


