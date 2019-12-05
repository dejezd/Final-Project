##
# Data Wrangler for EMG Classification RNN
# Jeremy Decker
# 11/14/19
# This Program will create a series of entries by gesture in each file in the data, to be further processed in the
# model itself.
##

import pandas as pd
import numpy as np
import glob
import os

#print(os.listdir('data/'))


def main():
    # To do, make it so that this program processes all 72 datasets.
    files_path = 'Data/'
    read_files = glob.glob(os.path.join(files_path, "*.txt"))
    i = 0
    labels =[]
    # Initialize empty arrays to take on samples
    for file in read_files:
        emg_data = pd.read_csv(file, header=0, sep='\t')  # Read in an individual file
        print(emg_data.head())
        pemg = process(emg_data)  # Label each class change as a separate gesture
        #print(pemg)
        label = sample_gen(pemg, i)  # Generate output files and a vector of classes
        # Create a super list of labels for use in the RNN
        if i == 0:
            labels = label
        else:
            labels = np.append(labels, label)
        i = i + 1
    # Orient the Labels the way that I want it to
    labels = np.transpose(labels)
    np.savetxt("PData/emg_labels.csv", labels, delimiter=',')  # Save to a csv file

    #emg1 = pd.read_csv("data/1_raw_data_13-12_22.03.16.txt", header=0, sep='\t')
    #print(emg1.head())
    #emg1 = emg1.to_numpy()
    #print(emg1[1, :])


def process(data):
    """
    This function will sort the data by identifying individual gestures and label them for further processing
    :param data: the data set to be labeled and exported as new files.
    :return: table - The table with each gesture individually labeled.
    """
    tdata = data
    table = np.c_[tdata, (np.zeros(np.size(tdata, 0)))]
    #print(table)
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

def sample_gen(data, fnum):
    """
    This function will write files based on the set label, after eliminating unmarked data (Class 0)
    :param data: The data set processed using the process function above, this will allow us to parse the dataset to
                 what it needs to be to export and be used in the model.
    :param fnum: This will define what the origin file of each sample, ordered from top to bottom from the Data Folder
    :return: This function returns a 1x11 array of the class label for each gesture processed.
    """
    maxindex = np.size(data, 0) - 1
    k = 0  # Used in File Naming, to enumerate the gestures of a particular file
    m = 0
    label = np.zeros(1)
    for i in range(0, int(data[maxindex, 10])):
        table = []
        count = 0
        for j in range(0, np.size(data, 0)):
            if count == 0:
                if data[j, 9] != 0 and data[j, 10] == i:
                    table = np.r_[table, data[j, :]]
                    count = count + 1
                    if m == 0:
                        label[m] = data[j, 9]
                    else:
                        label = np.append(label, data[j, 9])
            elif data[j, 9] != 0 and data[j, 10] == data[j-1, 10]:
                table = np.c_[table, data[j, :]]
            else:
                m = m+1
                break
        if np.size(table, 0) != 0:
            print("Creating file for Sample ", fnum, "Number ", k)
            #print(table)
            table = table[1:9, :]
            table = np.transpose(table)
            np.savetxt("PData/emg_sample_%d_%d.csv" % (fnum, k), table, delimiter=',')
            k = k+1
    print(label)
    return label


if __name__ == '__main__':
    main()
