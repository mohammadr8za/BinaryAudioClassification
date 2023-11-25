import os
import torch
import pandas as pd
import torchaudio
import BinLoad
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class SoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 target_sample_rate,
                 num_samples,
                 device,
                 transformation=None
                 ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = self.annotations["dir"]
        self.device = device
        if transformation:
            self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Use torchaudio for wav or other format and use bin load for binary format
        # signal = TxtLoad.txt_load_single_mic(audio_sample_path)
        signal = BinLoad.bin_load(audio_sample_path)
        signal = signal.to(self.device)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # fold = f"{self.annotations.iloc[index, 2]}"
        path = os.path.join(self.audio_dir[index], self.annotations.iloc[index, 2],
                            self.annotations.iloc[index, 3])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 4]


class SoundDataset2(Dataset):
    # we can normalize input data with this dataset, we load all data and scale entire dataset sample features at once
    def __init__(
            self,
            annotations_file,
            scalers=None,
            feature_scaling=False,
            sample_scaling=False,
            global_normalization=False,
            valid_norm=False,
            test_norm=False,
            norm_param=None,
            Feature="MFCC"
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.feature_scaling = feature_scaling
        self.scalers = scalers
        self.sample_scaling = sample_scaling
        self.signals = []
        self.labels = []

        if self.sample_scaling and self.feature_scaling:
            raise ValueError(f"Both feature and sample scaling can't be"
                             f" true at same time : {self.sample_scaling}, {self.feature_scaling}")

        for index, row in tqdm(self.annotations.iterrows(), total=len(self.annotations.index)):
            # load all labels and signals
            audio_sample_path = os.path.join(row['dir'],
                                             row['fold'],
                                             row['sample_path'])
            if Feature == "MFCC":
                self.signals.append(BinLoad.bin_load_mfcc(audio_sample_path).squeeze(dim=0))
            if Feature == "S1":
                self.signals.append(BinLoad.bin_load_s1(audio_sample_path).squeeze(dim=0))
            self.labels.append(row['class'])

        self.signals = torch.stack(self.signals)
        if self.sample_scaling:
            # some custom scaling, just dividing by max
            data = self.signals

            if not scalers:
                max = data.max()
                self.scalers = max
            else:
                max = self.scalers

            self.signals = data / max

        if global_normalization:

            self.mean = self.signals.mean()
            self.std = self.signals.std()
            # self.std = 21.85

            self.signals = (self.signals - self.mean) / self.std

        if valid_norm:

            self.mean = norm_param[0]
            self.std = norm_param[1]

            self.signals = (self.signals - self.mean) / self.std

        if test_norm:

            self.mean = torch.tensor(norm_param["mean"])
            self.std = torch.tensor(norm_param["std"])

            self.signals = (self.signals - self.mean) / self.std


        if self.feature_scaling:
            # applying min max scaling to all features
            data = self.signals.detach().cpu().numpy()
            # self.signals = data.detach().cpu().numpy()

            if not scalers:
                self.scalers = {}
                for i in range(self.signals.shape[1]):
                    self.scalers[i] = MinMaxScaler(feature_range=(0, 1))
                    self.scalers[i].fit(data[:, i, :])
                    data[:, i, :] = self.scalers[i].transform(data[:, i, :])
            else:
                self.scalers = scalers
                for i in range(self.signals.shape[1]):
                    data[:, i, :] = self.scalers[i].transform(data[:, i, :])

            self.signals = torch.from_numpy(data)

        # mobilenet accepts image like data, add a channel dim
        self.signals = self.signals.unsqueeze(dim=1)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        x = self.signals[index]

        return x, self.labels[index]


class train_test_valid:
    def __init__(self, data):
        self.data = data
        # self.batch_size = batch_size

    def get_data_set(self):
        data_train, data_test1 = train_test_split(self.data, test_size=.2, shuffle=True, random_state=42)
        data_test, data_valid = train_test_split(data_test1, test_size=.4, shuffle=True, random_state=42)
        # data_loader_train = DataLoader(data_train, self.batch_size)
        # data_loader_test = DataLoader(data_test, self.batch_size)
        # data_loader_valid = DataLoader(data_valid, self.batch_size)

        return data_train, data_test, data_valid


if __name__ == '__main__':

    DATA_VERSION = 'DATA_VERSION1'
    data_train = SoundDataset2(
        fr"D:/mreza/TestProjects/Python/BinaryAudioClassification/Data/{DATA_VERSION}/Data_Train_Annotation.csv",
        # feature_scaling=True,
        sample_scaling=False,
        global_normalization=True
    )
    train_sample, _ = data_train[0]
