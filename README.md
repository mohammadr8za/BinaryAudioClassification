# BinaryAudioClassification
Binary classification of audio sound using Conformer and AST models based on MFCC features

## DATA
Data required for training/testing is stored in binary files (.bin) and each file comprises 80 MFCC coefficients corresponding to 5 consequtive audio/speech frames (16 MFCC in each frame). In case you prefer to change the storation type, then modifications in the Custom Dataset will also be required. So, it is recommended to save your data in binary files to prevent requiring further modifications. 
After stroring data in binary files they must be placed in the [Data] folder with the specific dataset ID or Version (e.g., DATA_VERSION1). 

**NOTE1:** You may create a number of datasets and place them into the [Data] folder. 

**NOTE2:** your data must be placed in two different folders defining two classes for classification. 

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
You may also modify hyperparameters of the training algorithm using the following commands:

| Hyperparameter (type) | Command | 
| ------------- | ------------- |
| Dataset root (string) | --dataset_root |
| Dataset ID/Version (string) | -d / --dataset |
| Learning Rate (float) | -lr |
| Batch Size (list of integers) | -bs |
| Dropout (list of floats) | -drop |
| Gamma (list of floats) | -g |
| Epochs (integer) | -e / --epoch |
| Save Root (string) | -s / --save_root |
| Device (choices) | --device |
| Model (choices) | --model |

## Outputs
After running the training/testing program, its results will be strored in the [Experiments] folder with a specific ID which includes date and configuration of each particular run. Results include saving model in .pt format, confusion matrix, and loss/accuracy figures. Beside normalization parameters regarding the dataset used will bes stored in the folder related to each run with a .csv file.


## Conformer Inclusion
To include Conformer model in addition to the AST model follow the instructions mentioned below:

clone [@confomrer](https://github.com/mohammadr8za/conformer.git) repository into your project directory:
```
cd YOUR_LOCAL_DIRECTORY/BinaryAudioClassification
git clone https://github.com/mohammadr8za/conformer.git
```
then, in Terminal:

```
python new_model.py --model 'conformer'
```


## Author
* Mohammad Reza Peyghan [@mohammadr8za](https://github.com/mohammadr8za)
* Contact mohammdreza.peyghan@gmail.com


