import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
import Custom_Model
from Custom_data_set import SoundDataset, SoundDataset2
from sklearn.metrics import confusion_matrix
from utils import Normalization
from conformer.conformer import Conformer

torch.manual_seed(4321)
torch.cuda.manual_seed(4321)
np.random.seed(4321)
random.seed(4321)

# if mfcc dimensions has changed you must change Binload, input_fdim, inpuf_tdim and ast model img_size
device = 'cuda' if torch.cuda.is_available else 'cpu'
NUM_OF_CLASS = 1
INPUT_TDIM = 16
INPUT_FDIM = 5
LEARNING_RATE = 1e-6
Batch_size = [32]
SAMPLE_RATE = 16000
NUM_SAMPLES = 80
DATA_VERSION = 'DATA_VERSION1'
MODEL_ID = 'myModelID'

# MODEL_CHECKPOINT = 'best_model.pt'
BEST_EPOCH = 204
MODEL_CHECKPOINT = 'epoch_'+str(BEST_EPOCH)+'.pt'
EXPERIMENTS_SAVE_ROOT = Path(r'D:\mreza\TestProjects\Python\BinaryAudioClassification\Experiments\Test')
EXPERIMENTS_LOAD_ROOT = Path(r'D:\mreza\TestProjects\Python\BinaryAudioClassification\Experiments\Train\All_in')
EXPERIMENT_ID = f"test_[{DATA_VERSION}]_" \
                f"[{'_'.join(MODEL_ID.split('_')[1:])}]_" \
                f"{MODEL_CHECKPOINT.split('.')[0]}"

feature = 'MFCC'
in_dim = 23

data_scalers = {'max': 97.77}

data_test = SoundDataset2(
    fr"D:\mreza\TestProjects\Python\BinaryAudioClassification\Data\{DATA_VERSION}\Data_test_Annotation.csv",
    feature_scaling=False,
    sample_scaling=False,
    test_norm=False,
    global_normalization=True,
    Feature=feature
)

data_loader_test = DataLoader(data_test, Batch_size[0])

# TODO: You may alternate it with AST model
model = Conformer(num_classes=2, input_dim=in_dim, encoder_dim=32, num_encoder_layers=12).to(device)

checkpoint = torch.load(EXPERIMENTS_LOAD_ROOT / MODEL_ID / 'checkpoints' / MODEL_CHECKPOINT)
state_dict = checkpoint['net']
model.load_state_dict(state_dict)

truelabels = []
predictions = []


def test_model():
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader_test):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.squeeze().repeat(1, 1, 2)
        inputs = inputs.reshape(-1, int(inputs.shape[1] * 2), int(inputs.shape[2] / 2))

        outputs = model(inputs.type(torch.float))

        for label in targets:
            truelabels.append(label.item())

        for prediction in torch.argmax(outputs.squeeze(), dim=1):
            predictions.append(prediction.item())

    save_path = EXPERIMENTS_SAVE_ROOT / EXPERIMENT_ID / 'reports'
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Plot the confusion matrix
    cm = confusion_matrix(truelabels, predictions)
    df_cm = pd.DataFrame(cm, index=['NON_OK', 'OK'],
                         columns=['NON_OK', 'OK'])

    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig(save_path / r"confusion_numerical.png")
    plt.close()

    cm = confusion_matrix(truelabels, predictions, normalize='true')
    df_cm = pd.DataFrame(cm, index=['NON_OK', 'OK'],
                         columns=['NON_OK', 'OK'])

    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig(save_path / r"confusion_percent.png")
    plt.close()


test_model()
