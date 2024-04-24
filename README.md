# FLAD: Adaptive Federated Learning for DDoS Attack Detection

FLAD (a Federated Learning approach to DDoS Attack Detection) is an adaptive Federated Learning (FL) approach for training feed-forward neural networks, that implements a mechanism to monitor the classiﬁcation accuracy of the global model on the clients’ validations sets, without requiring any exchange of data. Thanks to this mechanism, FLAD can estimate the performance of the aggregated model and dynamically tune the FL process by assigning more computation to those clients whose attacks proﬁles are harder to learn.

More details on the architecture of FLAD and its performance in terms of detection accuracy and execution time are available in the following research paper:

Roberto Doriguzzi-Corin, Domenico Siracusa, "FLAD: Adaptive Federated Learning for DDoS attack detection", in Computers & Security,
Volume 137, 2024, doi: 10.1016/j.cose.2023.103597


## Installation

The current FLAD's Framework is implemented in Python v3.9 and tested with Keras and Tensorflow 2.7.1. It inherits the traffic pre-processing tool from the LUCID project (https://github.com/doriguzzi/lucid-ddos), also implemented in Python v3.9 with the support of Numpy and Pyshark libraries. 

FLAD requires the installation of a number of Python tools and libraries. This can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).
We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

Execute one of the following commands (based on your operating system) and follow the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh (on Linux operating systems)
bash Miniconda3-latest-MacOSX-x86_64.sh (on macOS)
```

Then create a new ```conda``` environment (called ```python39```) based on Python 3.9:

```
conda create -n python39 python=3.9
```

Activate the new ```python39``` environment:

```
conda activate python39
```

And configure the environment with ```tensorflow``` and a few more packages:

On Linux operating systems:
```
(python39)$ pip install tensorflow==2.7.1
(python39)$ pip install scikit-learn h5py pyshark protobuf==3.19.6
```

On macOS (tested on Apple M1 CPU)
```
(python39)$ conda install -c conda-forge tensorflow=2.7.1
(python39)$ conda install -c conda-forge scikit-learn h5py pyshark
```
Pyshark is used in the ```lucid_dataset_parser.py``` script for data pre-processing.
Pyshark is just Python wrapper for tshark, meaning that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current parser code works with ```tshark``` **version 3.2.3 or lower** or **version 3.6 or higher**. Issues have been reported when using intermediate releases such as 3.4.X.

For the sake of simplicity, we omit the command prompt ```(python39)$``` in the following example commands in this README.   ```(python39)$``` indicates that we are working inside the ```python39``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

FLAD has been tested with a Multi Layer Perceptron (MLP) model consisting of two fully connected hidden layers of 32 neurons each. The output layer includes a single neuron whose value represents the predicted probability of a traffic flow of being malicious (DDoS). The input is an array-like representation of a traffic flow, the same implemented in LUCID. For this reason, FLAD adopts the same traffic preprocessing tool of LUCID, including support to the CIC-DDoS2019 DDoS dataset from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html). Follows the same procedure for traffic pre-processing presented in the LUCID repository (https://github.com/doriguzzi/lucid-ddos).

FLAD requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid_dataset_parser.py``` Python script. It currently supports three DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, plus a custom dataset containing a SYN Flood DDoS attack (SYN2020). FLAD has been tested on the CIC-DDoS2019 dataset and the results are reported in the paper referenced above. More information on this traffic pre-processing tool can be found in the LUCID documentation.


### First step

The traffic pre-processing operation comprises two steps. The first parses the file with the labels (if needed) all extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped into flows, where a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result is a set of intermediate binary files with extension ```.data```.

This first step can be executed with the following command:

```
python3 lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder /path_to/dataset_folder/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-DOS2019-preprocess.data```. In general, the script loads all the ```pcap``` files contained in the folder indicated with option ```--dataset_folder``` and starting with prefix ```dataset-chunk-```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pre-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```DOS2019``` in the filename is the result of option ```--dataset_id DOS2019``` in the command.

Time window and flow length are two hyperparameters of FLAD and LUCID. For more information, please refer to the research paper mentioned above or the original LUCID paper (10.1109/TNSM.2020.2971776). 

### Second step

The second step loads the ```*.data``` files, merges them into a single data structure stored in RAM memory,  balances the dataset so that number of benign and DDoS samples are approximately the same, splits the data structure into training, validation and test sets, normalises the features between 0 and 1 and executes the padding of samples with zeros so that they all have the same shape.

Finally, three files (training, validation and test sets) are saved in *hierarchical data format* ```hdf5``` . 

The second step is executed with the command:

```
python3 lucid_dataset_parser.py --preprocess_folder /path_to/dataset_folder/
```

If the option ```--output_folder``` is not used, the output will be produced in the input folder specified with option ```--preprocess_folder```.

At the end of this operation, the script prints a summary of the pre-processed dataset. In our case, with these tiny traffic traces, the result should be something like the following:
```
2022-02-18 07:05:54 | examples (tot,ben,ddos):(402,202,200) | Train/Val/Test sizes: (321,37,44) | Packets (train,val,test):(2201,252,310) | options:--preprocess_folder /path_to/dataset_folder/00-WebDDoS/ |
```
This means 402 samples in total (202 benign and 200 DDoS), 321 in the training set, 37 in the validation set and 44 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid_dataset_parser.py``` script is saved within the output folder in the ```history.log``` file.

## Evaluation

FLAD's main script is ```flad_main.py```. The script implements a range of tests to train an NN model (either a Multi-Layer Perceptron (MLP) or a Convolutional Neural Network (CNN)) under federated learning settings, and to compare FLAD against state-of-the-art FL solutions such as: (i) the Federated Averaging algorithm ([FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html)) proposed by McMahan et al. in Communication-efﬁcient learning of deep networks from decentralized data, and (ii) Federated Learning DDoS ([FLDDoS](https://www.computer.org/csdl/proceedings-article/cbd/2022/074500a236/1EVijQfCe7m)), which has been designed to tackle the issues of FedAvg associated with non-i.i.d. DDoS traffic data. 

### The training process

The federated training process can be started by executing the following command:
```
python3 flad_main.py --clients /path_to/client_folders/ 
```

where option ```--clients``` (or ```-c```) is used to indicate the folder that contains the clients' local datasets. The folder must be organised in subfolders, one for each client, containing the local training and validation sets in ```.hdf5``` format.

By default, the script executes the FL procedure using the FLAD algorithms, including the adaptive selection and tuning of the clients. While the automated mechanism for the selection of clients cannot be disabled when using the "FLAD" training mode, you have the flexibility to specify a fixed number of local epochs and steps per epoch for each client in every round by using one of the following command-line listed below. 

In addition, the ```flad_main.py``` script can be configured to train the global model either with FedAvg or FLDDoS. The command-line options supported by ```flad_main.py``` are the following:
- ```-c```, ```--clients```: folder with the clients local datasets.
- ```-t```, ```--training_mode```: FL mode. Options are: flad, flddos, fedavg_1e, fedavg_5e. 
- ```-o```, ```--output_folder```: folder where FLAD saves the final global model and the federated training log files (default: ```./log```). If the folder does not exist, it will be automatically created.
- ```-e```, ```--local_epochs```: number of local training epochs performed by clients at every round of federated training (default: ```None```, which means adaptive).
- ```-s```, ```--steps_per_epoch```: number of local MBGD steps/epoch performed by clients at every round of federated training (default: ```None```, which means adaptive).
- ```-m```, ```--model```: Load a model from disk (path to a ```hd5``` file) or use a predefined NN architecture (```mlp``` (default) and ```cnn``` are possible options).
- ```-r```, ```--rounds```: Number of rounds of federated training. Ignored when ```flad``` training model is used (in that case, a *patience*-based mechanism is always used).
- ```-O```, ```--optimizer```: Optimizer used by clients to train the global model (options are SGD and Adam).
- ```-S```, ```--rn_seed```: Seed for the pseudo-number generators of various libraries used in the code, such as: Tensorflow, Numpy, Scikit-learn, etc.).

The final global model and the log of the training process are saved in ```h5``` and ```csv``` format respectively in the folder specified with the option ```--output_folder```, or within a subfolder of folder ```./log```, if ```--output_folder``` is not used.

### Paper's tests
The experiments presented in the aforementioned FLAD paper can be reproduced by executing the commands listed below.

#### Convergence analysis
In this test, we compare FLAD against FedAvg and FLDDoS in terms of convergence time and accuracy of the global model. For the experiments in the paper, the test has been repeated 10 times, each time with a different seed (option ```-rn_seed```).
First, execute with the ```flad_main.py``` script with option ```-t flad``` (FLAD settings) and specify a random seed with option ```--rn_seed```. The command to do that is the following:
```
python flad_main.py -c ./Dataset/CIC-DDoS2019 -t flad -o log/full-training/flad/test_seed0 --rn_seed 0
```
Then, note the number of rounds taken to reach convergence, and train again with the other approaches (flddos, fedavg_1e, fedavg_5e) till that number of rounds (```-r``` option) with the following commands:
```
for test in fedavg_1e fedavg_5e flddos; do python flad_main.py -c ./Dataset/CIC-DDoS2019 -t ${test} -o log/full-training/${test}/test_seed0 -r 62 --rn_seed 0; done

```
Please note that the above commands are executed 10 times, each time with a different random seed. The second command specifies ```-r 62``` which, in our tests, is the number of rounds taken by ```flad``` to reach convergence (25 rounds of patience included). Please change that number according to the results obtained on your testing machine.

#### Retraining test
This experiment starts with two clients training on attack data (one attack type each) and benign traffic. Once the federated training process converges, the resulting aggregated model is used as a starting point for another round of federated training, in which we provide attack data (a new attack type) to a third client to simulate the discovery of a new zero-day DDoS attack. Once convergence is achieved, we restart training by introducing new attack data on a fourth client. This is repeated until all thirteen attacks have been added, one on each client, and all the clients are provided with a model that has been trained with all the attack profiles.
In our evaluation, we tested with the following command:
```
for i in 2 3 4 5 6 7 8 9 10 11 12 13; do python flad_main.py -c Dataset/Retraining/test${i}/ -o Dataset/Retraining/test${i}/ -m Dataset/Retraining/test$((i-1))/10t-10n-mlp-global-model.h5  -t flad; done
```

Where ```test${i}``` is the folder that contains a federation of ```i``` clients, with ```$i=2,3,4,5,6,7,8,9,10,11,12,13```. Please note that at iteration ```i```, the script loads the model produced at iteration ```i-1```. At the first iteration, with```i=2```, the script generates a new ```mlp``` model.

#### Scalability test
In this test, we assess FLAD’s performance across a range of increasing federation sizes, from 13 to 90 clients.
We start with a federation of 13 clients (one-to-one mapping between clients and the attacks of the CICDDoS2019 dataset) and we progressively augment it with new clients generated by selecting two local datasets from the original set of clients in various combinations. This process can produce up to 78 new clients, each with a distinct local dataset, as calculated by the formula ```n!/(k!(n − k)!)``` with n = 13 and k = 2, for a maximum federation size of ```13 + 78 = 91``` clients.
In the paper, we tested with the following command:
```
for i in 13 20 30 40 50 60 70 80 90; do python flad_main.py -c Dataset/CIC-DDoS2019_${i}_clients -t flad; done
```
The above command iterates over a range of folders, each containing a federation of ```i``` number of clients, with ```i={13,20,30,40,50,60,70,80,90}```.


## Acknowledgements

If you are using FLAD's code for scientific research, please cite the related paper in your manuscript as follows:

Roberto Doriguzzi-Corin, Domenico Siracusa, "FLAD: Adaptive Federated Learning for DDoS attack detection", in Computers & Security,
Volume 137, 2024, doi: 10.1016/j.cose.2023.103597

This work has been partially supported by the European Union's Horizon Europe Programme under grant agreement No 101070473 (project FLUIDOS). 


## License

The code is released under the Apache License, Version 2.0.

