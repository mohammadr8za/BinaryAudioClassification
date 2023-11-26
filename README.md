# BinaryAudioClassification
Binary classification of audio sound using Conformer and AST models based on MFCC features

## DATA
Data required for training/testing is stored in binary files (.bin) and each file include 80 coefficient corresponding to 5 consequtive audio/speech frames (16 MFCC in each frame). In case you prefer to change storation type, then modifications in the Custom Dataset will also be required. So, it is recommended to save your data in binary files to prevent further modifications. 

## How to run?
First clone the repository in YOUR_LOCAL_DIRECTORY:

```
cd [YOUR_LOCAL_DIRECTORY]
git clone https://github.com/mohammadr8za/BinaryAudioClassification.git
```
then, run the code using following commands:

```
cd BinaryAudioClassification
python new_model.py --dataset_root [YOUR_LOCAL_DIRECTORY/BinaryAudioClassification/Data] -d [DATASET_ID/DATASET_VERSION] 
```
You may also modify hyperparameters of the training with the following commands:



## Conformer Inclusion
To include Conformer model in addition to the AST model follow the instructions mentioned below:

clone [@confomrer](https://github.com/mohammadr8za/conformer.git) repository into your project directory:
```
cd ../BinaryAudioClassification
git clone https://github.com/mohammadr8za/conformer.git
```
then, in Terminal:

```
python new_model.py --model 'conformer'
```


## Author
* Mohammad Reza Peyghan [@mohammadr8za](https://github.com/mohammadr8za)
* Contact mohammdreza.peyghan@gmail.com


