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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tensorflow log level
from util_functions import *
import tensorflow as tf
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, LSTM, Reshape
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Layer
from tensorflow.keras.models import Model, Sequential, save_model, load_model, clone_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow._api.v2.math import reduce_sum, square
tf.keras.utils.set_random_seed(SEED)
K.set_image_data_format('channels_last')

# disable GPUs for test reproducibility
tf.config.set_visible_devices([], 'GPU')


KERNELS = 256
MLP_UNITS = 32

MIN_EPOCHS = 1
MAX_EPOCHS = 5
MIN_STEPS = 10
MAX_STEPS = 1000

def compileModel(model, optimizer_type="SGD",loss='binary_crossentropy'):
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    else:
        optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])

# Convolutional NN
def CNNModel(model_name, input_shape,kernels,kernel_rows,kernel_col,classes=1, pool_height='max',regularization=None,dropout=None):
    K.clear_session()

    model = Sequential(name=model_name)
    if regularization == 'l1' or regularization == "l2":
        regularizer = regularization
    else:
        regularizer = None

    model.add(Conv2D(kernels, (kernel_rows,kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Activation('relu'))
    current_shape = model.layers[0].output_shape
    current_rows = current_shape[1]
    current_cols = current_shape[2]
    current_channels = current_shape[3]

    # height of the pooling region
    if pool_height == 'min':
        pool_height = 3
    elif pool_height == 'max':
        pool_height = current_rows
    else:
        pool_height = 3

    pool_size = (min(pool_height, current_rows), min(3, current_cols))
    model.add(MaxPooling2D(pool_size=pool_size, name='mp0'))
    model.add(Flatten())
    model.add(Dense(classes, activation='sigmoid', name='fc1'))

    print(model.summary())
    return model

# MPL model
def FCModel(model_name, input_shape, units, classes=1,dropout=None):
    K.clear_session()

    model = Sequential(name=model_name)

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(units, activation='relu', name='fc0'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu', name='fc1'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Dense(classes, activation='sigmoid', name='fc3'))

    print(model.summary())
    return model

def init_server(model_type, dataset_name, input_shape, max_flow_len):
    server = {}
    server['name'] = "Server"
    features = input_shape[1]

    if model_type == 'cnn':
        server['model'] = CNNModel(dataset_name + "-CNN", input_shape, kernels=KERNELS, kernel_rows=min(3,max_flow_len), kernel_col=features)
    elif model_type == 'mlp':
        server['model'] = FCModel(dataset_name + "-MLP", input_shape, units=MLP_UNITS)
    elif model_type is not None:
        try:
            server['model'] = load_model(model_type)
        except:
            print("Error: Invalid model file!")
            return None
    else:
        print("Error: Please use option \"model\" to indicate a model type (mlp or cnn), or to provide a pretrained model in h5 format")
        return None

    return server
    

def init_client(subfolder, X_train, Y_train, X_val, Y_val, dataset_name, time_window, max_flow_len):
    client = {}
    client['name'] = subfolder.strip('/').split('/')[-1] #name of the client based on the folder name
    client['folder'] = subfolder
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    client['training'] = (X_train_tensor,Y_train)
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    client['validation'] = (X_val_tensor,Y_val)
    client['samples'] = client['training'][1].shape[0]
    client['dataset_name'] = dataset_name
    client['input_shape'] = client['training'][0].shape[1:4]
    client['features'] = client['training'][0].shape[2]
    client['classes'] =  np.unique(Y_train)
    client['time_window'] = time_window
    client['max_flow_len'] = max_flow_len
    reset_client(client)
    return client

def reset_client(client):
    client['local_model'] = None # local model trained only with local data (FLDDoS comparison)
    client['f1_val'] = 0         # F1 Score of the current global model on the validation set
    client['f1_val_best'] = 0    # F1 Score of the best model on the validation set
    client['loss_train'] = float('inf')
    client['loss_val'] = float('inf')
    client['epochs'] = MIN_EPOCHS
    client['steps_per_epoch'] = MIN_STEPS
    client['rounds'] = 0
    client['round_time'] = 0
    client['update'] = True

def check_clients(clients):
    input_shape = clients[0]['input_shape']
    features = clients[0]['features']
    classes = clients[0]['classes']
    time_window = clients[0]['time_window']
    max_flow_len = clients[0]['max_flow_len']
    for client in clients:
        if input_shape != client['input_shape'] or \
            features != client['features'] or \
            classes.all() != client['classes'].all() or \
            time_window != client['time_window'] or \
            max_flow_len != client['max_flow_len']:
                print("Inconsistent clients properties!")
                return False
    return True