import torch
import torch.nn as nn
import jdatetime
import seaborn as sns
import numpy as np
import random
import pandas as pd
import logging
from pathlib import Path
from os.path import join
import engine_new_model
from Custom_data_set import SoundDataset2
import Custom_Model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import wandb
import engine
import utils
from torch.utils.tensorboard import SummaryWriter
from conformer.conformer.model import Conformer
import argparse

torch.manual_seed(4321)
torch.cuda.manual_seed(4321)
np.random.seed(4321)
random.seed(4321)

# Parameter Initialization
Feature = "S1"

SAMPLE_RATE = 16000
NUM_SAMPLES = 80  # target sample rate
EPOCHS = 10
LEARNING_RATE = 1e-03
NUM_OF_CLASS = 1
BATCH_SIZE = [2048]
n_fft = 128
TRESHOLD = .5
N_LFCC = 256
TRANSFORMATION = "mfcc"
INPUT_TDIM = 16
INPUT_FDIM = 5
DROPOUT = [0.000005]
GAMMA = [0.985]    # learning rate decay
# EXTRA = 'weighted_scaled_sample_normalized_wd9e-6'
EXPERIMENTS_SAVE_ROOT = Path(r'D:\mreza\TestProjects\Python\BinaryAudioClassification\Experiments')
EXPERIMENT_TIMESTAMP = '{0:%Y%m%d}'.format(jdatetime.datetime.now())
DATASETS = ['DATA_VERSION1']


def list_of_strings(arg):
    return list(arg.split(','))

# A R G P A R S E
parser = argparse.ArgumentParser(description='Model_Training')
parser.add_argument("--dataset_root", dest="dataset_root", type=str,
                    default=r"D:\mreza\TestProjects\Python\BinaryAudioClassification\Data",
                    help="Define the root directory including dataset with their IDs")
parser.add_argument('-d', '--dataset', dest='datasets', type=str, default=DATASETS, nargs='+',
                    help='Define dataset directory')
parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-05, help='Define a Learning Rate')
parser.add_argument('-drop', dest='dropout', default=[0], type=float, nargs='+', help='Define Dropout Rate')
parser.add_argument('-bs', dest='batch_size', default=[64], type=int, nargs='+', help='Define Batch Size')
parser.add_argument('-g', dest='gamma', default=[0.985], type=float, nargs='+', help='Define Gamma')
parser.add_argument('-e', '--epochs', dest='epochs', default=3, type=int, help='Define Number of Epochs')
parser.add_argument("-s", "--save_root", dest="save_root", default=EXPERIMENTS_SAVE_ROOT, type=Path,
                    help="Define a specific directory to save the experimental results")

# Device Agnostic Code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser.add_argument('--device', dest='device', default=device, choices=['cuda', 'cpu'],
                    help="Choose the device to run your code ['cuda', 'cpu'] (Default: cuda is available)")

parser.add_argument('--model', dest='model', default='ast', choices=['conformer', 'ast'],
                    help="Choose the model you want to train ['conformer', 'ast']")

args = parser.parse_args()

# wandb.login(key='1b15ee7e724f6c9ea5102d02e7c6a2dc5c0b72a7')


class Checkpoint(object):

    def __init__(self):
        self.best_acc = 0.
        self.dir = args.save_root / EXPERIMENT_ID / 'checkpoints'
        if not self.dir.exists():
            self.dir.mkdir(parents=True)

    def save(self, acc, epoch, filename=''):
        engine.logger.info('Saving checkpoint...')

        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = self.dir / f'epoch_{epoch}.pt'
        torch.save(state, save_path)
        # self.best_acc = acc


config = {"lr": args.learning_rate,
          "decay": 0,
          "dropout": 0,  # it will get initialized in the "Dropout loop"
          "batch_size": 0,
          "epochs": args.epochs,
          'dataset_id': 0}


results_dict = {"DATA": [], "#Epochs": [], "LearningRate": [], "Decay": [], "BatchSize": [],
                "Dropout": [], "TrainAcc": [], "ValidAcc": [], "BestEpoch": [], "TrainTime": []}

scalers = 100


for dataset in args.datasets:

    Feature = "MFCC"
    input_dim = 16

    config['dataset_id'] = dataset
    print(f"_________________________ {config['dataset_id']} _________________________")

    data_train = SoundDataset2(
        join(args.dataset_root, config['dataset_id'], "Data_Train_Annotation.csv"),
        sample_scaling=False,
        global_normalization=True,
        norm_param=None,
        Feature=Feature
    )
    data_scalers = data_train.scalers
    print(f"Data Scaler: {data_scalers}")

    normalization_parameters = torch.zeros((2, ))
    normalization_parameters[0], normalization_parameters[1] = data_train.mean, data_train.std
    norm_param_dict = {"mean": [normalization_parameters[0].item()], "std": [normalization_parameters[1].item()]}
    # norm_param_dict = {"mean": [0], "std": [0]}
    df_norm = pd.DataFrame(norm_param_dict)
    df_norm.to_csv("norm_param" + "_" + "DATA_VERSION")

    data_valid = SoundDataset2(
        join(args.dataset_root, config['dataset_id'], "Data_valid_Annotation.csv"),
        sample_scaling=False,
        global_normalization=True,
        valid_norm=False,
        Feature=Feature
    )

    INPUT_TDIM = data_train.signals.shape[-1]
    INPUT_FDIM = data_train.signals.shape[-2]

    for g in range(len(args.gamma)):
        config['decay'] = GAMMA[g]
        print(f"_________________________ GAMMA: {config['decay']} _________________________")

        for b in range(len(args.batch_size)):
            config["batch_size"] = args.batch_size[b]
            print(f"_________________________ BATCH_SIZE: {config['batch_size']} _________________________")

            for d in range(len(args.dropout)):  # Dropout loop
                config["dropout"] = args.dropout[d]

                # wandb_config_dict["dropout"] = DROPOUT[d]
                # wandb.init(
                #     project="AST_Model",
                #     config=wandb_config_dict
                # )
                #
                # config = wandb.config
                print(f"_________________________ DROPOUT: {config['dropout']} _________________________")

                EXPERIMENT_ID = f"{EXPERIMENT_TIMESTAMP}_{config['dataset_id']}_{TRANSFORMATION}_{INPUT_FDIM}_{INPUT_TDIM}" \
                                f"_BS{config['batch_size']}_LR{config['lr']}_G{config['decay']}_D{config['dropout']}_normalization_Conformer_7block"

                # DataLoader
                data_loader_train = DataLoader(data_train, config['batch_size'], shuffle=True)
                # data_loader_test = DataLoader(data_test, config['batch_size'])
                data_loader_valid = DataLoader(data_valid, config['batch_size'])

                if args.model == 'ast':

                    # Define the model (+drop_out) (+MODEL SIZE: Choose from ['tiny224', 'small224', 'base224', 'base384'])
                    ast_mdl = Custom_Model.model(input_tdim=INPUT_TDIM, input_fdim=INPUT_FDIM, label_dim=NUM_OF_CLASS,
                                                       drop_out=config['dropout'], model_size='small224').get_model()
                    # state_dict = torch.load(r"D:\mreza\TestProjects\Python\BinaryAudioClassification\Experiments\Train\All_in\myModelID.pt")
                    # ast_mdl.load_state_dict(state_dict["net"])
                    # ast_mdl.to(device)
                    model = ast_mdl

                if args.model == 'conformer':
                    model = Conformer(num_classes=2, input_dim=input_dim, encoder_dim=32, num_encoder_layers=7)
                    # state_dict = torch.load(
                    #     r"D:\mreza\TestProjects\Python\BinaryAudioClassification\Experiments\Train\All_in\myModelID.pt")
                    # model.load_state_dict(state_dict["net"])

                model.to(args.device)

                # initialise loss funtion + optimiser
                label_counts = torch.bincount(data_train.labels)
                weights = torch.tensor(
                    [len(data_train.labels) / label_counts[0], len(data_train.labels) / label_counts[1]])
                loss_fn = nn.CrossEntropyLoss(weight=weights.half().to(device))
                # Define Regularization-included
                optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=9e-6)
                # optimiser = torch.optim.SGD(ast_mdl.parameters(), lr=config['lr'], weight_decay=9e-5, nesterov=True,
                #                             momentum=0.9)
                scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=config['decay'])

                checkpoint = Checkpoint()

                train_loss = []
                valid_loss = []
                train_accuracy = []
                valid_accuracy = []

                best_model = {'val_acc': 0,
                              'train_acc': 0,
                              'epoch': 0}
                writer = SummaryWriter(comment=f"LR{config['lr']}G{config['decay']}D{config['dropout']}"
                                               f"BS{config['batch_size']}E{config['epochs']}ID{config['dataset_id']}")

                # Save Normalization Parameters
                df_norm.to_csv(join(args.save_root, EXPERIMENT_ID, "norm_param.csv"))

                tic = timer()
                print(f"_________________________ Device: {device} _________________________")
                for epoch in range(config["epochs"]):
                    print(f"\n_______________ Epoch: {epoch + 1} _______________\n")
                    print(
                        f"DATA_ID: {config['dataset_id']} | LR: {LEARNING_RATE} | GAMMA: {config['decay']} | BATCH_SIZE: {config['batch_size']} | DROPOUT: {config['dropout']} | Device: {device}\n")
                    TrainLoss, TrainAccuracy = engine_new_model.train(epoch=epoch, dataloader=data_loader_train,
                                                                      model=model, optimizer=optimiser,
                                                                      loss_fn=loss_fn, device=device)
                    Test_Loss, TestAccuracy = engine_new_model.valid(epoch=epoch, dataloader=data_loader_valid,
                                                                     model=model, loss_fn=loss_fn,
                                                                     device=device, checkpoint=checkpoint)
                    train_loss.append(TrainLoss)
                    train_accuracy.append(TrainAccuracy)
                    valid_loss.append(Test_Loss)
                    valid_accuracy.append(TestAccuracy)
                    # Learning Rate Decay
                    epoch_lr = optimiser.param_groups[0]["lr"]
                    engine.logger.info(f'Learning Rate: {epoch_lr: .10f}')
                    scheduler.step()

                    # engine.result(EXPERIMENTS_SAVE_ROOT=EXPERIMENTS_SAVE_ROOT, EXPERIMENT_ID=EXPERIMENT_ID,
                    #               train_loss=train_loss, valid_loss=valid_loss, train_accuracy=train_accuracy,
                    #               valid_accuracy=valid_accuracy, data_loader_test=data_loader_test, model=ast_mdl,
                    #               device=device)

                    engine_new_model.result(EXPERIMENTS_SAVE_ROOT=args.save_root, EXPERIMENT_ID=EXPERIMENT_ID,
                                            train_loss=train_loss, valid_loss=valid_loss, train_accuracy=train_accuracy,
                                            valid_accuracy=valid_accuracy, data_loader_test=data_loader_valid, model=model,
                                            device=device)

                    best_model = utils.save_best_model(val_acc=TestAccuracy, train_acc=TrainAccuracy,
                                                       epoch=epoch, best_model=best_model,
                                                       address=args.save_root / EXPERIMENT_ID)

                    # wandb.log({"Valid Acc": TestAccuracy, "Valid Loss": Test_Loss,
                    #            "Train Acc": TrainAccuracy, "Train Loss": TrainLoss})
                    writer.add_scalars(main_tag="loss",
                                       tag_scalar_dict={"train loss": TrainLoss,
                                                        "test loss": Test_Loss},
                                       global_step=epoch)
                    writer.add_scalars(main_tag="Accuracy",
                                       tag_scalar_dict={"train acc": TrainAccuracy,
                                                        "test acc": TestAccuracy},
                                       global_step=epoch)

                toc = timer()
                training_time = int(toc - tic)
                print(f"Training time on {device}: {training_time} seconds")
                # wandb.finish()
                writer.close()
                config_dict = utils.csv_best_configs_board(config=config, best_model=best_model,
                                                           config_dict=results_dict, time=training_time,
                                                           address=args.save_root)



