import torch
from os.path import join
import pandas as pd


def save_best_model(val_acc, train_acc, epoch, best_model, address):

    if val_acc >= best_model['val_acc']:
        best_model['val_acc'] = val_acc
        best_model['train_acc'] = train_acc
        best_model['epoch'] = epoch
    else:
        pass

    torch.save(obj=best_model, f=join(address, 'best_model.pt'))

    return best_model


def csv_best_configs(config, best_model, config_dict, time, address):

    config_dict["DATA"].append(config.dataset_id)
    config_dict["#Epochs"].append(config.epochs)
    config_dict["LearningRate"].append(config.lr)
    config_dict["Decay"].append(config.decay)
    config_dict["BatchSize"].append(config.batch_size)
    config_dict["Dropout"].append(config.dropout)
    config_dict["TrainAcc"].append(best_model["train_acc"])
    config_dict["ValidAcc"].append(best_model["val_acc"])
    config_dict["BestEpoch"].append(best_model["epoch"])
    config_dict["TrainTime"].append(time)

    dataframe = pd.DataFrame(config_dict)
    dataframe.to_csv(join(address, 'results.csv'), index=False)

    return config_dict

def csv_best_configs_board(config, best_model, config_dict, time, address):   # for mytrain_tensorboard

    config_dict["DATA"].append(config['dataset_id'])
    config_dict["#Epochs"].append(config['epochs'])
    config_dict["LearningRate"].append(config['lr'])
    config_dict["Decay"].append(config['decay'])
    config_dict["BatchSize"].append(config['batch_size'])
    config_dict["Dropout"].append(config['dropout'])
    config_dict["TrainAcc"].append(best_model["train_acc"])
    config_dict["ValidAcc"].append(best_model["val_acc"])
    config_dict["BestEpoch"].append(best_model["epoch"])
    config_dict["TrainTime"].append(time)

    dataframe = pd.DataFrame(config_dict)
    dataframe.to_csv(join(address, 'results_board.csv'), index=False)

    return config_dict


def Normalization(input_data, type="sample"):
    # Local Normalizaton
    if type == "feature":
        mean = torch.mean(input_data, 2)
        std = torch.std(input_data, 2)

        input_data = (input_data - mean.unsqueeze(2)) / std.unsqueeze(2)

    if type == "sample":
        mean = torch.mean(input_data, 3)
        std = torch.std(input_data, 3)

        input_data = (input_data - mean.unsqueeze(3)) / std.unsqueeze(3)

    return input_data


