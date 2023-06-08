# Copyright (c) 2023 @ FBK - Fondazione Bruno Kessler
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
import random as rn

def load_dataset(path):
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])  # features
    set_y_orig = np.array(dataset["set_y"][:])  # labels

    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig#.reshape((1, set_y_orig.shape[0]))

    return X_train, Y_train

def load_set(folder_path,set_type,seed):
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
        files = glob.glob(dataset_folder + "/*" + '-' + set_type + '.hdf5')

        for file in files:
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

            set_list.append(load_dataset(file))

    # Concatenation of all the training and validation sets
    X = set_list[0][0]
    Y = set_list[0][1]
    for n in range(1, len(set_list)):
        X = np.concatenate((X, set_list[n][0]), axis=0)
        Y = np.concatenate((Y, set_list[n][1]), axis=0)

    X, Y = shuffle(X, Y,random_state=seed)

    return X,Y, time_window, max_flow_len, dataset_name

def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)