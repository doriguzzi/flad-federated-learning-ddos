# FLAD: Adaptive Federated Learning for DDoS Attack Detection

FLAD (a Federated Learning approach to DDoS Attack Detection) is an adaptive Federated Learning (FL) approach for training feed-forward neural networks, that implements a mechanism to monitor the classiﬁcation accuracy of the global model on the clients’ validations sets, without requiring any exchange of data. Thanks to this mechanism, FLAD can estimate the performance of the aggregated model and dynamically tune the FL process by assigning more computation to those clients whose attacks proﬁles are harder to learn.

More details on the architecture of FLAD and its performance in terms of detection accuracy and execution time are available in the following research paper:

R. Doriguzzi-Corin and D. Siracusa, "FLAD: Adaptive Federated Learning for DDoS Attack Detection," in *arXiv preprint arXiv:2205.06661*, doi: 10.48550/arXiv.2205.06661, 2022.


## Installation

The current FLAD's Framework is implemented in Python v3.9 with Keras and Tensorflow 2.7. It inherits the traffic pre-processing tool  from the LUCID project (https://github.com/doriguzzi/lucid-ddos), also implemented in Python v3.9 with the support of Numpy and Pyshark libraries. 

FLAD requires the installation of a number of Python tools and libraries. This can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).
We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

In a Linux OS, execute the following command and follows the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a new ```conda``` environment (called ```python39```) based on Python 3.9 and including part the required packages:

```
conda create -n python39 python=3.9 numpy tensorflow=2.7.0 h5py lxml
```

Activate the new ```python39``` environment:

```
conda activate python39
```

And finalise the installation with a few more packages:

```
(python39)$ pip3 install pyshark sklearn seaborn
```
In particular, Pyshark is used in the ```lucid_dataset_parser.py``` script of the [LUCID project] (https://github.com/doriguzzi/lucid-ddos) for data pre-processing.
Pyshark is just Python wrapper for tshark, meaning that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current parser code works with ```tshark``` **version 3.2.13 or lower**. Issues have been reported when using newest releases such as 3.4.X.

For the sake of simplicity, we omit the command prompt ```(python39)$``` in the following example commands in this README.   ```(python39)$``` indicates that we are working inside the ```python39``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

FLAD has been tested with a Multi Layer Perceptron (MLP) model consisting of two fully connected hidden layers of 32 neurons each. The output layer includes a single neuron whose value represents the predicted probability of a traffic flow of being malicious (DDoS). The input is an array-like  representation of a traffic flow, the same implemented in LUCID. For this reason, FLAD adopts the same traffic prepocessing tool of LUCID, including the support to the CIC-DDoS2019 DDoS dataset from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html). Follows the same procedure for traffic pre-processing presented in the LUCID repository (https://github.com/doriguzzi/lucid-ddos).

FLAD requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid_dataset_parser.py``` Python script of the [LUCID project] (https://github.com/doriguzzi/lucid-ddos). It currently supports three DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, plus a custom dataset containing a SYN Flood DDoS attack (SYN2020). FLAD has been tested on the CIC-DDoS2019 dataset and the results are reported in the paper referenced above. More information on this traffic pre-processing tool can be found in the LUCID documentation.


### First step

The traffic pre-processing operation comprises two steps. The first parses the file with the labels and extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped in flows, where a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result is a set of intermediate binary files with extension ```.data```.

This first step can be executed with command:

```
python3 lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder /path_to/dataset_folder/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-DOS2019-preprocess.data```. In general, the script loads all the ```pcap``` files contained in the folder indicated with option ```--dataset_folder``` and starting with prefix ```dataset-chunk-```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pre-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```DOS2019``` in the filename is the result of option ```--dataset_id DOS2019``` in the command.

Time window and flow length are two hyperparameters of FLAD and LUCID. For more information, please refer to the research paper mentioned above or the original LUCID paper (10.1109/TNSM.2020.2971776). 

### Second step

The second step loads the ```*.data``` files, merges them into a single data structure stored in RAM memory,  balances the dataset so that number of benign and DDoS samples are approximately the same, splits the data structure into training, validation and test sets, normalises the features between 0 and 1 and executes the padding of samples with zeros so that they all have the same shape.

Finally, three files (training, validation and test sets) are saved in *hierarchical data format* ```hdf5``` . 

The second step is executed with command:

```
python3 lucid_dataset_parser.py --preprocess_folder /path_to/dataset_folder/
```

If option ```--output_folder``` is not used, the output will be produced in the input folder specified with option ```--preprocess_folder```.

At the end of this operation, the script prints a summary of the pre-processed dataset. In our case, with this tiny traffic traces, the result should be something like:

```
2022-02-18 07:05:54 | examples (tot,ben,ddos):(402,202,200) | Train/Val/Test sizes: (321,37,44) | Packets (train,val,test):(2201,252,310) | options:--preprocess_folder /path_to/dataset_folder/00-WebDDoS/ |
```

Which means 402 samples in total (202 benign and 200 DDoS), 321 in the training set, 37 in the validation set and 44 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid_dataset_parser.py``` script is saved within the output folder in the ```history.log``` file.

## Evaluation

FLAD's main script is ```flad_main.py```. The script implements a range of test to train the MLP model under federated learning settings, and to compare FLAD against the Federated Averaging algorithm (FedAvg) proposed by McMahan et al. in Communication-efﬁcient learning of deep networks from decentralized data, 2017 (http://proceedings.mlr.press/v54/mcmahan17a.html). 

### Command options

To execute the federated training process, the following parameters can be specified when using ```flad_main.py```:

- ```-t```, ```--train_federated```: Starts the federated training process and specifies the folder with the dataset. The folder must be organised in subfolders, one for each client.
- ```-e```, ```--full_training ```: Performs the federated training across all the clients, either with the ```flad``` or with the ```fedavg``` algorithm.
- ```-r```, ```--retraining ```: Performs a progressive federated training across the clients, either with the ```flad``` or with the ```fedavg``` algorithm. It starts with only two clients (hence, with only two attack types), and progressively adds a new attack type every time convergence is achieved.

### The training process

The experiments presented in the aforementioned FLAD paper can be reproduced by executing the following commands:

```
python3 flad_main.py --train_federated /path_to/dataset_folder/ --full_training flad
python3 flad_main.py --train_federated /path_to/dataset_folder/ --full_training fedavg

python3 flad_main.py --train_federated /path_to/dataset_folder/ --retraining flad
python3 flad_main.py --train_federated /path_to/dataset_folder/ --retraining fedavg

```

The results of the process are saved in ```csv``` format in a folder called ```log```, which is automatically created within the code's main folder when the training process is executed for the first time.



## Acknowledgements

If you are using FLAD's code for a scientific research, please cite the related paper in your manuscript as follows:

R. Doriguzzi-Corin and D. Siracusa, "FLAD: Adaptive Federated Learning for DDoS Attack Detection," in *arXiv preprint arXiv:2205.06661*, doi: 10.48550/arXiv.2205.06661, 2022.


## License

The code is released under the Apache License, Version 2.0.

