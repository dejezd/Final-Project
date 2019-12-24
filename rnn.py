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
from keras.preprocessing import sequence
import glob
import os


def main():
    files_name = 'PData/'
    print('Collecting Data...')
    fnames = glob.glob(os.path.join(files_name, 'emg_sample_*.csv'))
    print('Setting Testing and Training Datasets...')
    train_set, test_set = train_test_split(fnames, test_size=0.25, random_state=1)
    print('Processing Training Dataset...')
    labeled_tds, labels_train, maxentry_trn = process_data(train_set, True)
    print('Processing Testing Dataset...')
    labeled_test, labels_test, maxentry_tst = process_data(test_set, False)
    if maxentry_trn > maxentry_tst:
        maxent = maxentry_trn
    else:
        maxent = maxentry_tst

    labeled_tds = sequence.pad_sequences(labeled_tds, maxlen=maxent, dtype='float32', padding='post')
    labeled_test = sequence.pad_sequences(labeled_test, maxlen=maxent, dtype='float32', padding='post')

    model = gen_model()

    model.compile(optimizer=tf.keras.optimizers.Adamax(0.00025),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(labeled_tds, labels_train, epochs=25,
                        validation_data=(labeled_test, labels_test))
    model.save('lstm_model4.h5')

    test_loss, test_acc = model.evaluate(labeled_test, labels_test)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    # Insert Plotting functions here
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 0.75])
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
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
        tf.keras.layers.LSTM(64), # Will play around with more hyperparameters here.
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.125),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.125),
        tf.keras.layers.Dense(16, activation='relu'),
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
    maxentry = 0
    i = 0
    for f in file_path:
        file = np.loadtxt(f, delimiter=',')
        data = file[:, 0:8]  # Should input all 8 columns of data needed
        data_all[i] = data
        if np.size(data, 0) > maxentry:
            maxentry = np.size(data, 0)
        label = file[1, 8] - 1
        label_all[i] = label
        i = i+1
    data_all = np.asarray(data_all)
    #print('Dataset: ', data_all)
    #print('Labels: ', label_all)
    print(maxentry)
    return data_all, label_all, maxentry


if __name__ == '__main__':
    main()
