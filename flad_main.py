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

import sys
from flad_training import *
import argparse

def main(argv):
    help_string = 'Usage: python3 flad_main.py --train_federated <dataset_folder> --retraining flad'

    parser = argparse.ArgumentParser(
        description='FLAD, Adaptive Federated Learning for DDoS Attack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train_federated', nargs='?', type=str,
                        help='Start the federated training process')

    parser.add_argument('-e', '--epochs', nargs='+',default=None, type=int,
                        help='Training iterations')

    parser.add_argument('-s', '--steps_per_epoch',default=None, type=int,
                        help='Steps of gradient descent taken at each epoch')

    parser.add_argument('-ro', '--rounds', default=0, type=int,
                        help='Federated training iterations')

    parser.add_argument('-o', '--optimizer', type=str, default="SGD",
                        help='Optimizer (SGD, Adam)')

    parser.add_argument('-of', '--output_folder', nargs='?', type=str, default="./log",
                        help='Folder which stores the training/testing results (default: ./log ')

    parser.add_argument('-ft', '--full_training', default="flad", type=str, nargs='+',
                        help='Federated training mode. Available options are (multiple choices are permitted): flad, fedavg')
    parser.add_argument('-rt', '--retraining', default="flad", type=str, nargs='+',
                        help='Federated training mode. Available options are (multiple choices are permitted): flad, fedavg')


    args = parser.parse_args()

    if os.path.isdir(args.output_folder) == False:
        os.mkdir(args.output_folder)

    if args.train_federated is not None:
        subfolders = glob.glob(args.train_federated + "/*/")
        subfolders.sort()

        # clients initialisation
        clients = []
        for subfolder in subfolders:
            try:
                X_train, Y_train, time_window, max_flow_len, dataset_name = load_set(subfolder, "train")
                X_val, Y_val, time_window, max_flow_len, dataset_name = load_set(subfolder, "val")
            except:
                continue

            client = init_client(subfolder, X_train, Y_train, X_val, Y_val, dataset_name, time_window, max_flow_len)
            clients.append(client)

        if len(clients) == 0:
            print("No clients found!")
            exit(-1)

        # check clients consistency
        if check_clients(clients) == False:
            exit(-1)

        # test with progressive introduction of new attacks
        for test in args.full_training:
            if test == "flad":
                FederatedTrain(clients, 'mlp', args.output_folder, time_window, max_flow_len, dataset_name,
                               epochs_list=['auto'], steps_list=['auto'], training_clients_list=["flad"], weighted=False,
                               optimizer=args.optimizer, nr_experiments=EXPERIMENTS)
            # test training a rundom subset of clients and using the mcmahan's paper paramters (E=1,5, B=50)
            elif test == "fedavg":
                FederatedTrain(clients, 'mlp', args.output_folder, time_window, max_flow_len, dataset_name,
                               epochs_list=[1, 5], steps_list=[0], training_clients_list=["fedavg"], weighted=True,
                               optimizer=args.optimizer, nr_experiments=EXPERIMENTS)

        # test with progressive introduction of new attacks
        for test in args.retraining:
            if test == "flad":
                FederatedReTrain(clients, 'mlp', args.output_folder, time_window, max_flow_len, dataset_name,
                               epochs_list=None, steps_list=None, training_clients_list=["flad"], weighted=False,
                               optimizer=args.optimizer,nr_experiments=EXPERIMENTS)
            # test training a rundom subset of clients and using the mcmahan's paper paramters (E=1,5, B=50)
            elif test == "fedavg":
                FederatedReTrain(clients, 'mlp', args.output_folder, time_window, max_flow_len, dataset_name,
                               epochs_list=[1,5], steps_list=[0], training_clients_list=["fedavg"],  weighted=True,
                               optimizer=args.optimizer,nr_experiments=EXPERIMENTS)


if __name__ == "__main__":
    main(sys.argv[1:])
