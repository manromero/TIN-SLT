# Explore More Guidance: A Task-aware Instruction Network for Sign Language Translation Enhanced with Data Augmentation

This repository fork of the orignal [tin-slt](https://github.com/yongcaoplus/tin-slt), has been created with the aim of supporting the study of intelligent models for sign language translation (SLT), which is part of the Master's Thesis (TFM) carried out for the [Master in Computer Engineering (MII)](https://www.mii.us.es/) of the [University of Seville](https://www.us.es/).

Originally created by [Yong Cao](https://yongcaoplus.github.io/), Wei Li, [Xianzhi Li](https://nini-lxz.github.io/), [Min Chen](https://people.ece.ubc.ca/~minchen/), [Guangyong Chen](https://guangyongchen.github.io/), Zhengdao Li, [Long Hu](https://people.ece.ubc.ca/~minchen/longhu/), [Kai Hwang](https://myweb.cuhk.edu.cn/hwangkai).

## 1. Introduction

 This repository is for our Findings of NAACL 2022 paper '[Explore More Guidance: A Task-aware Instruction Network for Sign Language Translation Enhanced with Data Augmentation
](https://arxiv.org/abs/2204.05953)'. In this paper, we propose a task-aware instruction network, namely TIN-SLT, for sign language translation, by introducing the instruction module and the learning-based feature fuse strategy into a Transformer network. 
 In this way, the pre-trained model's language ability can be well explored and utilized to further boost the translation performance. 
 Moreover, by exploring the representation space of sign language glosses and target spoken language, we propose a multi-level data augmentation scheme to adjust the data distribution of the training set. 
 We conduct extensive experiments on two challenging benchmark datasets, PHOENIX-2014-T and ASLG-PC12, on which our method outperforms former best solutions by 1.65 and 1.42 in terms of BLEU-4.


## 2. Dataset and Trained models
* Dataset can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1NNta7CgBF0Ny5IbzmBKP-C1B33aR6kWZ?usp=sharing).       
* The original trained model can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1s26goE0Rh4T9L_d-6XDfHKYP_FPGPveR?usp=sharing). 
If the trained model doesn't work or if there are any issues, please feel free to contact us.
* The original pre-trained model can be downloaded in [bert-base-german-dbmdz-uncased](https://drive.google.com/file/d/105RuAqXLYj5mPeHbp3pzEKCWlSL7wR0o/view?usp=sharing).
* Best model found after performing hyperparameter scanning: [our best TIN-SLT](https://uses0-my.sharepoint.com/:u:/g/personal/mignunolm_alum_us_es/EaraWClmZkNFr29i3YfcDX8By3JW98IYSDSdFCL6kU_6wg?e=Xf8xBe)
## 3. Execute Steps

May differs from the original in [tin-slt](https://github.com/yongcaoplus/tin-slt):

#### Step 1 prepare environment

Clone the repository

```
git clone https://github.com/manromero/TIN-SLT
cd TIN-SLT
```

Create a new virtual environment using python 3.6

```
# create a new virtual environment using python 3.6
virtualenv --python=python3 venv
# If you have more than one python, you can specify the python file. 
# ex: `virtualenv --python=C:\Users\migu1\AppData\Local\Programs\Python\Python36\python.exe venv`
# Activate linux:
source venv/bin/activate
# Activate windows:
.\venv\Scripts\activate
```

#### Step 2 install dependencies

```shell script
pip install --editable .      
```

Note that, if the download speed is not fast, try this:

```shell script
pip install --editable . -i https://pypi.tuna.tsinghua.edu.cn/simple   
```

Verify that torch has been installed correctly

```
python
> import torch
> torch.cuda.is_available()
# True -> The installation has been successfully completed and it is possible to use the graphics card for training.
# False -> Despite a successful installation, it will not be possible to make use of the graphics card during training, which will cause errors during training.
```

If false:

1. Make sure you have configured CUDA and CUDNN correctly. An example configuration for Windows 11 is available [here](https://youtu.be/OEFKlRSd8Ic?t=123).
2. Perform the Torch installation using the commands available from the [official PyTorch website](https://pytorch.org/get-started/locally/), removing the installed version beforehand.

Originally the code is implemented over Python 3.6.8, and Pytorch 1.5.0+cu101. (It wasn't test on other package version.)      
```shell script
pip uninstall torch
pip install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Step 3 Prepare dataset

For facilitate the organization, prepare the next folder structure:
* Create folder "dataset" in the root of the directory
* Create folder "raw" and "processed" in the folder "dataset" created

Extract the google drives dataset in the previous folders.

If you use our prepared dataset, skip these steps. 

Configure "preprocessing/prepare_data.py" with the dataset path you are going to use.

Else:
```shell script
cd preprocessing      
python prepare_data.py
```
Then you can get the following files in your destination dir:
![](asset/prepare_data.png)


#### Step 4 Train by AutoML
Choose one training method (with / without automl).

**Important:** "Train without AutoML" was the method used to train the model in the research work carried out for the Master's thesis. We leave the rest of the instructions from the original article, although these have not been tested and may require additional instructions.
##### (1) Without AutoML

For Linux users:

```shell script
cd trainer
# please config train.sh first, and then:
sh train.sh
```

For Windows users:

*This script has been created with respect to the original documentation to make it easier for Windows users to train the model.*

```shell script
cd trainer
# please config train_windows.py first, and then:
python train_windows.py
```

Runtime: 2 hours 52 min (Approximate, using NVIDIA GeForce RTX 3070).

##### (2) With AutoML

**Note:** Not tested for the Master's Thesis.

Config automl/config.yml and automl/search_space.json files, and run the following cmd to train in your terminal:
```shell script
nnictl create --config automl/config.yml -p 11111
# -p means the port you wish to visualize the training process in browser.
```
If succeed, you should see the following logs in your terminal:
![](asset/train_start.png)

Go to your browser to see the training process.
![](asset/nni_plat.png)

![](asset/train_process.png)

Please refer to [NNI Website](https://nni.readthedocs.io/) for more instructions on NNI.


#### Step 5 Evaluate

For Linux users:

```shell script
cd postprocessing       
sh get_bleu4.sh
```

For Windows users:

*This script has been created with respect to the original documentation to make it easier for Windows users to evaluate the model.*

```shell script
cd postprocessing
# please config get_bleu4_stmc_windows.py first, and then:
python get_bleu4_stmc_windows.py
```

## 4. Unified flow and hyperparameter scanning

To facilitate the execution of the flow in a unified way, while enabling hyperparameter scanning, the python file "scan.py" has been created (Not available in the original repository). In it we can configure as a grid the combinations of hyperparameters that we want to test. Once configured, the complete training, inference and evaluation flow will be executed for each of the combinations set.

```
python scan.py
```

Once we have found the model with the best metrics, we can refine the selection of the hyperparameter "beam size" using the "beam_search.py" script.

```
python beam_search.py
```


## 5. Questions
Please contact [yongcao_epic@hust.edu.cn]().



## 6. Some problems you may encounter:

1.During dependencies installation: "Cannot open include file: 'basetsd.h': No such file or directory"

Install [winsdksetup](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)

1.'ascii' codec can't decode byte 0xef
```shell script
UnicodeDecodeError: 'ascii' codec can't decode byte 0xef in position 1622: ordinal not in range(128)
```
=> please run this command in your terminal

```shell script
export LC_ALL=C.UTF-8
source ~/.bashrc
```

2.Resource punkt not found. / Resource wordnet not found.

please run this command in your terminal
```shell script
python
  >>> import nltk
  >>> nltk.download('wordnet')
  >>> nltk.download('punkt')
```
