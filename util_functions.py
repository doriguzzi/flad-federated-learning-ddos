# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: FLAD, Adaptive Federated Learning for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import h5py
import glob
import scipy.stats
from collections import OrderedDict
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn

SEED = 0
# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

MAX_FLOW_LEN = 100 # number of packets
TIME_WINDOW = 10
TRAIN_SIZE = 0.90 # size of the training set wrt the total number of samples

protocols = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
powers_of_two = np.array([2**i for i in range(len(protocols))])

num_classes_2017 = 7
attacks2017 = ["Benign","DoS","FTP Patator","SSH Patator","Port Scan", "Botnet","Web Attack"]

num_classes_2019 = 2
attacks2019 = ["Benign","DDoS"]


# feature list with min and max values
feature_list = OrderedDict([
    ('timestamp', [0,10]),
    ('packet_length',[0,1<<16]),
    ('highest_layer',[0,1<<32]),
    ('IP_flags',[0,1<<16]),
    ('protocols',[0,1<<len(protocols)]),
    ('TCP_length',[0,1<<16]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_flags',[0,1<<16]),
    ('TCP_window_size',[0,1<<16]),
    ('UDP_length',[0,1<<16]),
    ('ICMP_type',[0,1<<8])]
)

def load_dataset(path):
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])  # features
    set_y_orig = np.array(dataset["set_y"][:])  # labels

    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig#.reshape((1, set_y_orig.shape[0]))

    return X_train, Y_train

def load_set(folder_path,set_type):
    set_list = []
    time_window = 0
    max_flow_len = 0
    dataset_name = 0
    subfolders = glob.glob(folder_path + "/*/")
    if len(subfolders) == 0:  # for the case in which the is only one folder, and this folder is args.train[0]
        subfolders = [folder_path + "/"]
    else:
        subfolders = sorted(subfolders)
    for dataset_folder in subfolders:
        dataset_folder = dataset_folder.replace("//", "/")  # remove double slashes when needed
        file = glob.glob(dataset_folder + "/*" + '-' + set_type + '.hdf5')[0]

        filename = file.split('/')[-1].strip()
        tw = int(filename.split('-')[0].strip().replace('t', ''))
        mfl = int(filename.split('-')[1].strip().replace('n', ''))
        dn = filename.split('-')[2].strip()
        if time_window == 0:
            time_window = tw
        else:
            if tw != time_window:
                print("Mismatching time window size among datasets!")
                return None
        if max_flow_len == 0:
            max_flow_len = mfl
        else:
            if mfl != max_flow_len:
                print("Mismatching flow length size among datasets!")
                return None
        if dataset_name == 0:
            dataset_name = dn
        else:
            if dn != dataset_name:
                print("Mismatching dataset type among datasets!")
                return None

        set_list.append(load_dataset(dataset_folder + "/*" + '-'+ set_type + '.hdf5'))

    # Concatenation of all the training and validation sets
    X = set_list[0][0]
    Y = set_list[0][1]
    for n in range(1, len(set_list)):
        X = np.concatenate((X, set_list[n][0]), axis=0)
        Y = np.concatenate((Y, set_list[n][1]), axis=0)

    X, Y = shuffle(X, Y, random_state=SEED)

    return X,Y, time_window, max_flow_len, dataset_name

def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def count_packets_in_dataset(X_list):
    packet_counters = []
    for X in X_list:
        TOT = X.sum(axis=2)
        packet_counters.append(np.count_nonzero(TOT))

    return packet_counters

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def all_same(items):
    return all(x == items[0] for x in items)

# min/max values of features based on the nominal min/max values of the single features (as defined in the feature_list dict)
def static_min_max(time_window=10):
    feature_list['timestamp'][1] = time_window

    min_array = np.zeros(len(feature_list))
    max_array = np.zeros(len(feature_list))

    i=0
    for feature, value in feature_list.items():
        min_array[i] = value[0]
        max_array[i] = value[1]
        i+=1

    return min_array,max_array

# min/max values of features based on the values in the dataset
def find_min_max(X,time_window=10):
    sample_len = X[0].shape[1]
    max_array = np.zeros((1,sample_len))
    min_array = np.full((1, sample_len),np.inf)

    for feature in X:
        temp_feature = np.vstack([max_array,feature])
        max_array = np.amax(temp_feature,axis=0)
        temp_feature = np.vstack([min_array, feature])
        min_array = np.amin(temp_feature, axis=0)

    # flows cannot last for more than MAX_FLOW_DURATION seconds, so they are normalized accordingly
    max_array[0] = time_window
    min_array[0] = 0

    return min_array,max_array

def normalize_and_padding(X,mins,maxs,max_flow_len,padding=True):
    norm_X = []
    for sample in X:
        if sample.shape[0] > max_flow_len: # if the sample is bigger than expected, we cut the sample
            sample = sample[:max_flow_len,...]
        packet_nr = sample.shape[0] # number of packets in one sample

        norm_sample = scale_linear_bycolumn(sample, mins, maxs, high=1.0, low=0.0)
        np.nan_to_num(norm_sample, copy=False)  # remove NaN from the array
        if padding == True:
            norm_sample = np.pad(norm_sample, ((0, max_flow_len - packet_nr), (0, 0)), 'constant',constant_values=(0, 0))  # padding
        norm_X.append(norm_sample)
    return norm_X

def padding(X,max_flow_len):
    padded_X = []
    for sample in X:
        flow_nr = sample.shape[0]
        padded_sample = np.pad(sample, ((0, max_flow_len - flow_nr), (0, 0)), 'constant',
                              constant_values=(0, 0))  # padding
        padded_X.append(padded_sample)
    return padded_X


def plot_distance_matrix(cm, attacks, features,outdir):
    color="Blues"

    xticks = features
    plot_height = 5

    # show the y labels only for the leftmost plots
    yticks = ""
    yticks = attacks
    plot_width = 5.1


    plt.figure(figsize=(plot_width, plot_height))
    #sns.set(font_scale=1.5)
    ax = sns.heatmap(cm, annot=True, cmap=color,fmt='.2f', xticklabels=xticks, yticklabels=yticks,cbar=False)
    #ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=16)
    #ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, horizontalalignment='right')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')


    #plt.title("JS Distance",fontsize=20)

    ax.figure.tight_layout()


    plt.savefig(outdir + "/distance_matrix.pdf")
    print("Saving the confusion matrix...")