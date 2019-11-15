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

#print(os.listdir('data/'))


def main():
    # To do, make it so that this program processes all 72 datasets.
    emg1 = pd.read_csv("data/1_raw_data_13-12_22.03.16.txt", header=0, sep='\t')
    print(emg1.head())
    print(np.size(emg1, 0))
    pemg1 = process(emg1)
    print(pemg1)


def process(data):
    """
    This function will sort the data by identifying individual gestures and label them for further processing
    :param data: the data set to be labeled and exported as new files.
    :return: table - The table with each gesture individually labeled.
    """
    tdata = data
    table = np.c_[tdata, np.zeros(np.size(tdata, 0))]
    print(table)
    j = 0
    for i in range(0, np.size(table, 0)):
        if i == 0:
            table[i, 10] = j
        elif table[i, 9] != table[i-1, 9]:
            j = j+1
            table[i, 10] = j
        else:
            table[i, 10] = j
    return table


def sample_gen(data):
    """
    This function will write files based on the set label, after eliminating unmarked data (Class 0)
    :param data: The data set processed using the process function above, this will allow us to parse the dataset to
                 what it needs to be to export and be used in the model.
    :return: This function does not have a return, as it saves files to a local directory instead.
    """
    for i in range(0, data[np.size(data, 0), 10]):
        print("Creating File for Gesture:", i)
        for j in range(0, np.size(data, 0)):
            if data[j, 10] != 0:
                if data[j, 10] == data[j-1, 10]:
                    table[i, :] = data[i, :]
        # INSERT THE SAVING AND NAMING FUNCTION HERE

        # Then save the table as an iterative name before moving on.
    return

# I need to write up a script that creates a new "small" table after the class changes and write the ones that are
# Non-zero into individual text files.
#Would it b easier to just mark it in another column? Probably.
if __name__ == '__main__':
    main()
