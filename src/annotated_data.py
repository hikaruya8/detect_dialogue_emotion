# -*- coding:utf-8 -*-

import re
import os
import csv
import argparse
import pickle
from tqdm import tqdm 
from itertools import chain
import numpy as np
import glob
from nltk.tokenize import RegexpTokenizer
import logging
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import data
from torchtext import vocab

train_data_txt = "../data/EMTC_data/emtc_train.txt"
train_data_csv = "../data/EMTC_data/emtc_train.csv"


def txt_to_csv():
    with open(train_data_txt, 'r') as fin, open(train_data_csv, 'w') as fout:
        o = csv.writer(fout)
        for line in fin:
            o.writerow(''.join(re.sub(':', '"', line, 1)).rsplit('#',1))

def cofirm_csv():
    # たしかめ
    with open(train_data_csv, 'r') as f:
        lines = f.readlines()
        print(lines)

def pandas_txt2csv():
    df = pd.read_csv(train_data_txt, header=None, sep=':|#', error_bad_lines=False, engine='python')
    df.to_csv(train_data_csv, index=False, header=None)
    print(df[:5])



def data_to_torch():
    tokenizer = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, fix_length=200)

    # LabelLField
    LABEL = data.LabelField()

    train_dataset = data.TabularDataset.splits(path='./', train=train_data_path, format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

    print('len(train)', len(train_dataset))
    # print('vars(train[0]', vars(train_dataset[0]))


if __name__ == '__main__':
    # txt_to_csv()
    # cofirm_csv()
    pandas_txt2csv()