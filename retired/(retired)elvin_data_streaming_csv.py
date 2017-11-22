import numpy as np
import pandas as pd
import csv
import os
import torch
import time
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SequentialSampler

root_path = '.'
train_file = '/'.join([root_path, 'full_train.csv'])
test_file = '/'.join([root_path, 'full_test.csv'])


def get_dataset_length(file_path, had_header=True):
    """Retrieve file length of a large csv file"""
    with open(file_path, 'r') as f:
        length = 0
        for _ in f:
            length += 1
        length = length - had_header
    return length

train_length = get_dataset_length(train_file)
test_length = get_dataset_length(test_file)


def get_col_names(file_path, had_header=True):
    if had_header:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, l in enumerate(reader):
                if i == 0:
                    header = l[1:]
                else:
                    break
        header = np.array(header)
    else:
        print("Your file does not have header.")
    return header

train_col_names = get_col_names(train_file)
test_col_names = get_col_names(test_file)


# build a Dataset that's iterable for csv
# TODO: csv iteration too time costly. create an alternative to load from
# sqlite
class KKBOXDataset(Dataset):
    """KKBOX Final Set for Training and Testing."""
    def __init__(self, csv_file, file_length, had_header=True):
        """
        Args:
        csv_file: path of the csv file
        file_length: integer of length of the csv file
        """
        self.file_path = csv_file
        self.file_length = file_length
        self.had_header = had_header

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        line = next(itertools.islice(
            csv.reader(open(self.file_path, 'r')), idx+self.had_header, None))
        line = np.array(line)
        line[line == ''] = '0.0'  # replace NaN with zero
        line = line.astype(float).reshape(1, -1)  # shape is (1, 133)
        return line  # iteration returns a (1, 133) numpy.ndarray

trainset = KKBOXDataset(train_file, train_length)
testset = KKBOXDataset(test_file, test_length)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=2000, shuffle=False,
                        sampler=testsampler)

start = time.time()
for i in trainloader:
    print(i)
    break


end = time.time()

'''Debug KKBOXDataset iterator working properly
for i in range(10):
    print(trainset[i].shape)
'''

'''Debug iterator logic here
with open('full_test.csv', 'r') as f:
    idx = 0
    line = next(itertools.islice(csv.reader(f), idx + True, None))
#    line = [x[:15] for x in line]
    line = np.array(line)
    line[line == ''] = '0.0'
    line = line.astype(float)
    print(line)
'''
