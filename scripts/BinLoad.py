import torch
import struct
import numpy as np


def bin_load_mfcc(path):
    with open(path, 'rb') as f:
        byte = f.read(8)
        counter = 8
        Data = np.array([])
        while byte:
            Data = np.append(Data, struct.unpack('d', byte))
            byte = f.read(8)
            counter += 8

    f.close()
    # change below lines if input shape has changed

    tensor = torch.tensor(Data).reshape(1, 5, 16)
    # tensor = torch.tensor(Data).reshape(1,5,16)[:,:,1:] # 0. remove first mfcc coefficient
    tensor[:, :, :1] /= 10 # 1. scaled first mfcc coefficient by 0.1

    return tensor

def bin_load_s1(path):
    with open(path, 'rb') as f:
        byte = f.read(8)
        counter = 8
        Data = np.array([])
        while byte:
            Data = np.append(Data, struct.unpack('d', byte))
            byte = f.read(8)
            counter += 8

    f.close()
    # change below lines if input shape has changed

    tensor = torch.tensor(Data).reshape(1, 5, 23)
    # tensor = torch.tensor(Data).reshape(1,5,16)[:,:,1:] # 0. remove first mfcc coefficient
    # tensor[:, :, :1] /= 10 # 1. scaled first mfcc coefficient by 0.1

    return tensor

