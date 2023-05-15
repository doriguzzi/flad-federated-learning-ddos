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

    parser.add_argument('-o', '--output_folder', nargs='?', type=str, default="./log",
                        help='Folder which stores the training/testing results (default: ./log ')
    
    parser.add_argument('-e', '--local_epochs', nargs='?', type=int, default=0,
                        help='Number of local epochs (default: 0, which means adaptive')


    args = parser.parse_args()

    if os.path.isdir(args.output_folder) == False:
        os.mkdir(args.output_folder)

    epochs = 'auto'
    if args.local_epochs >= 1:
        epochs = int(args.local_epochs)

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
        FederatedTrain(clients, 'mlp', args.output_folder, time_window, max_flow_len, dataset_name,
                               epochs=epochs, steps='auto', training_mode="flad", weighted=False,
                               optimizer="SGD", nr_experiments=EXPERIMENTS)


if __name__ == "__main__":
    main(sys.argv[1:])
