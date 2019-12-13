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


def main():
    #list_emg = tf.data.Dataset.list_files('PData/emg_sample_*.csv')
    #train_size = int(0.75*872)
    #list_emg = list_emg.shuffle(872)  # Shuffles data taking the whole dataset into account.
    #train = list_emg.take(train_size)
    #test = list_emg.skip(train_size)
    #for f in train.take(1):
     #   print(f.numpy())
        # print(tf.io.read_file(f))
    #labeled_tds = train.map(process_data(train, True))
    #labeled_test = test.map(process_data(test, False))
    files_name = 'PData/'
    fnames = glob.glob(os.path.join(files_name, 'emg_sample_*.csv'))
    train_set, test_set = train_test_split(fnames, test_size=0.25, random_state=1)
    train = tf.data.Dataset.from_tensor_slices(train_set)
    test = tf.data.Dataset.from_tensor_slices(test_set)
    labeled_tds = train.map(process_data(train_set, True))
    labeled_test = test.map(process_data(test_set, False))
    model = gen_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(labeled_tds, epochs=10,
                        validation_data=labeled_test,
                        validation_steps=30)
    test_loss, test_acc = model.evaluate(labeled_test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    # Insert Plotting functions here

# Start to set up the data for the sets
# Need to find a way to run the data for the model by reading the csv, and identifying the labels from it.
# Resource: https://datascience.stackexchange.com/questions/51249/training-keras-model-with-multiple-csv-files
# LSTM Start


def gen_model():
    """
    Creates the basic architecture for the LSTM
    :return: model, the initialized model for the LSTM RNN
    """
    print('Initializing Model...')
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # Will play around with more hyperparameters here.
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model


def process_data(file_path, train):
    """
    This processes the data to enter a tensorflow dataset. It sets the label to the 9th column,
    which indicates the gesture. The rest is put into the data section as a numpy array to be fed to the network.
    :param file_path: The files with which the
    :param train: Boolean, shows if data set size is the test or training sets.
    :return:
    """
    if train == True:
        size = 654
    else:
        size = 218
    # This might need a for loop to run through each file and return it to the map file?
    data_all = [None]*size
    label_all = np.zeros(size)
    i = 0
    for f in file_path:
        file = np.loadtxt(f, delimiter=',')
        data = file[0:8, :]  # Should input all 8 columns of data needed
        data_all[i] = data
        label = file[1, 8]
        label_all[i] = label
        i = i+1
    data_all = np.asarray(data_all)
    return data_all, label_all


if __name__ == '__main__':
    main()
