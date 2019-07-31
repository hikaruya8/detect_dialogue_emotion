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
from torchtext.vocab import GloVe

train_data_txt = "../data/EMTC_data/emtc_train.txt"
train_data_csv = "../data/EMTC_data/emtc_train.csv"
test_data_txt = "../data/EMTC_data/emtc_test.txt"
test_data_csv = "../data/EMTC_data/emtc_test.csv"


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
    train_data = pd.read_csv(train_data_txt, header=None, sep=':|#', error_bad_lines=False, engine='python')
    train_data.to_csv(train_data_csv, index=False, header=None)
    print(train_data[:5])

    test_data = pd.read_csv(test_data_txt, header=None, sep=':|#', error_bad_lines=False, engine='python')
    test_data.to_csv(test_data_csv, index=False, header=None)
    print(test_data[:5])



def data_to_torch():
    tokenizer = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, fix_length=50)
    LABEL = data.LabelField()

    # train_dataset = data.TabularDataset(path=train_data_csv, format='csv', fields=[('text', TEXT), ('label', LABEL)])
    # test_dataset = data.TabularDataset(path=test_data_csv, format='csv', fields=[('text', TEXT), ('label', LABEL)])

    train_dataset, test_dataset = data.TabularDataset.splits(path='../data/EMTC_data/', train='emtc_train.csv', test='emtc_test.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)])
    train_dataset, val_dataset = train_dataset.split()

    print('len(train): {}'.format(len(train_dataset)))
    print('len(validation): {}'.format(len(val_dataset)))    
    print('len(test): {}'.format(len(test_dataset)))

    # 単語に番号を振る
    TEXT.build_vocab(train_dataset, vectors=GloVe(name='840B', dim=300), min_freq=2)
    LABEL.build_vocab(train_dataset)

    # 単語カウント結果
    print('単語カウント: {}'.format(TEXT.vocab.freqs.most_common(10)))
    # print(LABEL.vocab.freqs)
    print('上位10個: {}'.format(TEXT.vocab.itos[:10])) #単語10個
    print('サイズ:{}'.format(TEXT.vocab.vectors.size()))

    # バッチ化
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset), batch_size=32, sort_key=lambda x: len(x.text), repeat=False,shuffle=True)

    vocab_size = len(TEXT.vocab)
    print(vocab_size)

    word_embeddings = TEXT.vocab.vectors
    print(TEXT.vocab.vectors.size())

    # 入力前のTensorの確認
    for batch in enumerate(train_iter):
        import pdb;pdb.set_trace()
        print(batch.text[0].size())
        print(batch.text[1].size())
        print(label.size())
        print("1データ目の単語列を表示")
        print(batch.text[0][0])
        print(batch.text[1][0])
        print(batch.label[0])
        print([TEXT.vocab.itos[data] for data in batch.text[0][0]].tolist())
        print('ラベル')
        print(batch.label[0].item())
        break



if __name__ == '__main__':
    # txt_to_csv()
    # cofirm_csv()
    # pandas_txt2csv()
    data_to_torch()