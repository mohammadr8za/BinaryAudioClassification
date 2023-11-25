from os import listdir
from os.path import join
import pandas as pd
from Custom_data_set import train_test_valid
import numpy as np
import argparse

DATA_VERSION = 'DATA_VERSION1'

parser = argparse.ArgumentParser(description="Make Data Annotation")

parser.add_argument("-dv", "--data_version", dest="data_version", default=DATA_VERSION, type=str,
                    help="Define Dataset Version to Make Its Annotation")
parser.add_argument("-r", "--root", dest="root", type=str,
                    default=r"D:\mreza\TestProjects\Python\BinaryAudioClassification\Data",
                    help="Define Dataset Directory")

args = parser.parse_args()

path = join(args.root, args.data_version)


def MakeMainAnnotation(mode, classtype="ok"):
    main_path = listdir(path)
    if mode == "main":

        ok_folders = listdir(join(path, main_path[0]))
        # Non_ok_folders = listdir(join(path, main_path[2], 'background'))
        Non_ok_folders = listdir(join(path, main_path[1]))
        data_non_ok = {"dir": [], "fold": [], "sample_path": [], "class": []}
        data_ok = {"dir": [], "fold": [], "sample_path": [], "class": []}
        for i in ok_folders:

            data_ok["dir"].append(path)
            data_ok["fold"].append(main_path[0])
            data_ok["sample_path"].append(i)
            data_ok["class"].append(1)

        for i in Non_ok_folders:

            data_non_ok["dir"].append(path)
            # data_non_ok["fold"].append(join(main_path[2], 'background'))
            data_non_ok["fold"].append(join(main_path[1]))
            data_non_ok["sample_path"].append(i)
            data_non_ok["class"].append(0)

        data_non_ok = pd.DataFrame(data_non_ok)
        data_ok = pd.DataFrame(data_ok)
        Data_Annotation = data_ok.append(data_non_ok)

        pd.DataFrame.to_csv(Data_Annotation, fr"{path}/Main_Data_Train_Annotation.csv")
    elif mode == 'inference':
        signal = listdir(join(path, main_path[4], 'single_mic_subsample'))
        signal_data = {"dir": [], "fold": [], "sample_path": [], "class": []}
        for i in signal:

            signal_data["dir"].append(join(path, main_path[4]))
            signal_data["fold"].append('single_mic_subsample')
            signal_data["sample_path"].append(i)
            if classtype == "ok":
                signal_data["class"].append(1)
            elif classtype == "non_ok":
                signal_data["class"].append(0)
        signal_data = pd.DataFrame(signal_data)
        pd.DataFrame.to_csv(signal_data, fr"{path}/Inference/multi_mic/Main_Data_Train_Annotation.csv")


def MakeSubAnnotation(mode):
    # Batch_size = batch_size
    if mode == 'main':
        Data_Annotation = pd.read_csv(fr"{path}\Main_Data_Train_Annotation.csv")
    elif mode == 'inference':
        Data_Annotation = pd.read_csv(fr"{path}\Inference\multi_mic\Main_Data_Train_Annotation.csv")

    Data_Annotation.drop('Unnamed: 0', axis=1, inplace=True)
    number_of_data = len(Data_Annotation)
    x_index = np.arange(0, number_of_data)
    train_index, test_index, valid_index = train_test_valid(x_index).get_data_set()

    train_annotation = Data_Annotation.loc[train_index.tolist(), :]
    test_annotation = Data_Annotation.loc[test_index.tolist(), :]
    valid_annotation = Data_Annotation.loc[valid_index.tolist(), :]

    # train_annotation= train_annotation.drop('Unnamed: 0', axis=1)
    # test_annotation= test_annotation.drop('Unnamed: 0', axis=1)
    # valid_annotation= valid_annotation.drop('Unnamed: 0', axis=1)
    if mode == "main":
        pd.DataFrame.to_csv(train_annotation,
                            fr"{path}\Data_Train_Annotation.csv")
        pd.DataFrame.to_csv(test_annotation,
                            fr"{path}\Data_test_Annotation.csv")
        pd.DataFrame.to_csv(valid_annotation,
                            fr"{path}\Data_valid_Annotation.csv")
    elif mode == "inference":
        pd.DataFrame.to_csv(train_annotation,
                            fr"{path}\Inference\multi_mic\Data_Train_Annotation.csv")
        pd.DataFrame.to_csv(test_annotation,
                            fr"{path}\Inference\multi_mic\Data_test_Annotation.csv")
        pd.DataFrame.to_csv(valid_annotation,
                            fr"{path}\Inference\multi_mic\Data_valid_Annotation.csv")


def MakeAnnForMainTrainData(main_data_path):
    main_path = listdir(main_data_path)
    train_data_path = join(main_data_path, main_path[1])
    test_data_path = join(main_data_path, main_path[0])
    valid_data_path = join(main_data_path, main_path[2])

    train_folders = listdir(train_data_path)
    test_folders = listdir(test_data_path)
    valid_folders = listdir(valid_data_path)

    data_sets = ['Train', 'Test', 'valid']
    data_sets_path = [train_data_path, test_data_path, valid_data_path]
    for i in range(len(data_sets)):

        ok_folder_path = join(data_sets_path[i], 'D')
        non_ok_folder_path = join(data_sets_path[i], 'N')
        ok_samples = listdir(ok_folder_path)
        non_ok_samples = listdir(non_ok_folder_path)
        data_non_ok = {"dir": [], "fold": [], "sample_path": [], "class": []}
        data_ok = {"dir": [], "fold": [], "sample_path": [], "class": []}

        for j in range(len(ok_samples)):
            data_ok["dir"].append(join(data_sets_path[i]))
            data_ok["fold"].append('D')
            data_ok["sample_path"].append(ok_samples[j])
            data_ok["class"].append(1)

        for j in range(len(non_ok_samples)):
            data_non_ok["dir"].append(join(data_sets_path[i]))
            data_non_ok["fold"].append('N')
            data_non_ok["sample_path"].append(non_ok_samples[j])
            data_non_ok["class"].append(0)

        data_non_ok = pd.DataFrame(data_non_ok)
        data_ok = pd.DataFrame(data_ok)
        Data_Annotation = data_ok.append(data_non_ok)
        # Data_Annotation = data_ok
        pd.DataFrame.to_csv(Data_Annotation, f"../Main_Data/{data_sets[i]}/Data_{data_sets[i]}_Annotation.csv")


if __name__ == "__main__":
    MakeMainAnnotation(mode='main')
    MakeSubAnnotation(mode="main")
    # MakeAnnForMainTrainData(f'../Main_Data')
