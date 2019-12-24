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
    print('Processing Testing Dataset...')
    labeled_test, labels_test, maxentry_tst = process_data(test_set, False)
    maxent = 4280
    print('Post-Padding Test Data...')
    labeled_test = sequence.pad_sequences(labeled_test, maxlen=maxent, dtype='float32', padding='post')

    print('Loading Model 1...')
    model = tf.keras.models.load_model('lstm_model2.h5')
    model.summary()
    loss, acc = model.evaluate(labeled_test, labels_test, verbose=2)
    print('Model 1 used a 0.0001 learning rate and had a dropout layer of 0.1')
    print('Model 1 Accuracy: {:5.2f}'.format(100*acc))
    print('Model 1 Loss: {:5}'.format(loss))

    print('Loading Model 2...')
    model2 = tf.keras.models.load_model('lstm_model4.h5')
    model2.summary()
    loss2, acc2 = model2.evaluate(labeled_test, labels_test, verbose=2)
    print('Model 2 Used a much larger model, starting with a 64 node LSTM and had two dropout layers.')
    print('Model 2 Accuracy: {:5.2f}'.format(100*acc2))
    print('Model 2 Loss: {:5}'.format(loss2))

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
