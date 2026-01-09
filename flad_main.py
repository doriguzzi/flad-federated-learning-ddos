# Copyright (c) 2026 @ FBK - Fondazione Bruno Kessler
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
    help_string = 'Basic usage: python flad_main.py -c Dataset/DOS2019_highly_unbalanced -t flad'

    parser = argparse.ArgumentParser(
        description='FLAD, Adaptive Federated Learning for DDoS Attack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--clients', nargs='?', type=str,
                        help='Folder with the clients local datasets')

    parser.add_argument('-e', '--local_epochs', nargs='?', type=int, default=None,
                        help='Number of local epochs (default: None, which means adaptive)')

    parser.add_argument('-m', '--model', type=str, default="mlp", choices=['mlp', 'cnn'],
                        help='Load model from disk (when a path is passed) or generate a new model (mlp and cnn are possible options)')

    parser.add_argument('-o', '--output_folder', nargs='?', type=str, default=None,
                        help='Folder which stores the training/testing results (default: ./log ')
    
    parser.add_argument('-s', '--steps_per_epoch', nargs='?', type=int, default=None,
                        help='Steps of gradient descent taken at each epoch (default: None, which means adaptive)')

    parser.add_argument('-O', '--optimizer', type=str, default="SGD", choices=['SGD', 'Adam'],
                        help='Optimizer (SGD, Adam)')
    
    parser.add_argument('-S', '--rn_seed', nargs='?', type=int, default=0,
                        help='Seed value for RNGs used in FLAD')

    args = parser.parse_args()

    SEED = args.rn_seed
    # Seed Random Numbers
    tf.keras.utils.set_random_seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)
    
    if args.output_folder == None:
        if os.path.isdir("./log") == False:
            os.mkdir("./log")
        output_folder = "./log" + "/federated_training-" + time.strftime("%Y%m%d-%H%M%S") + "/"
    else:
        output_folder = args.output_folder
        
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)

    # For the epochs, 0 has the same meaning than "None", i.e. "auto"
    epochs = 'auto'
    if args.local_epochs != None and args.local_epochs > 0:
        epochs = int(args.local_epochs)

    # For the steps, "None" means "auto", while 0 means computing the number of steps based on the batch_size
    steps = 'auto'
    if args.steps_per_epoch != None:
        steps = int(args.steps_per_epoch)

    if args.clients is not None:
        subfolders = glob.glob(args.clients + "/*/")
        subfolders.sort()

        # clients initialisation
        clients = []
        for subfolder in subfolders:
            try:
                X_train, Y_train, time_window, max_flow_len, dataset_name = load_set(subfolder, "train",SEED)
                X_val, Y_val, time_window, max_flow_len, dataset_name = load_set(subfolder, "val",SEED)
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

        # full FL training
        FederatedTrain(clients, args.model, output_folder, time_window, max_flow_len, dataset_name,
                        epochs=epochs, steps=steps, training_mode='flad', weighted=False,
                        optimizer=args.optimizer, nr_experiments=EXPERIMENTS)

if __name__ == "__main__":
    main(sys.argv[1:])
