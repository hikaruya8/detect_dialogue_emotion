# -*- coding: utf-8 -*-

import pickle
import numpy as np
from matplotlib import pyplot as plt
import copy
import argparse
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader

# ターミナルでパラメータを変えられるようにする
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--input_size", type=int, default=300)
parser.add_argument("--unsupervised_output_size", type=int, default=300) # unsupervisedでの出力次元
parser.add_argument("--supervised_output_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.001)
params = parser.parse_args()
print(params)

# confirm if we can use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def numpy2tensor():
    # numpy形式をtensorにする
    with open('input_vectors.pickle', 'rb') as f:
        input_vectors = pickle.load(f)

    torch_input_vectors = torch.FloatTensor(input_vectors).to(device)

    # import pdb;pdb.set_trace()

    with open('torch_input_vectors.pickle', 'wb') as f:
        pickle.dump(torch_input_vectors, f, protocol=4)


class Autoencoder(nn.Module): #nn.Moduleを継承
    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size) # fc = fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.75)
        self.unsuper_criterion = nn.MSELoss()

    def forward(self, x):
        x = F.sigmoid(self.dropout(self.fc1(x)))
        x = F.sigmoid(self.dropout(self.fc2(x)))
        x = F.sigmoid(self.dropout(self.fc2(x)))
        x = F.sigmoid(self.dropout(self.fc3(x)))

        return x


class SupervisedTrain(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output):
        super(SupervisedTrain, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.75)
        self.super_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.sigmoid(self.dropout(self.fc1(x)))
        x = F.sigmoid(self.dropout(self.fc2(x)))
        out = x
        return out


def train_autoencoder(net): # left part
    with open('torch_input_vectors.pickle', 'rb') as f:
        torch_input_vectors = pickle.load(f)

    train_loader = DataLoader(torch_input_vectors, batch_size=params.batch_size, shuffle=True)

    # import pdb;pdb.set_trace()

    unsuper_criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=params.learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)

    # save model and optimizer
    prev_net = copy.deepcopy(net)
    prev_optimizer = copy.deepcopy(optimizer)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in tqdm(range(params.num_epochs)):
        # エポックごとに初期化
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #訓練モードへ切り替え
        net.train()

        for i, data in enumerate(tqdm(train_loader)):
            net = net.to(device)
            # 順伝播の計算
            outputs = net(data)
            # lossの計算
            loss = unsuper_criterion(outputs, data)

            # lossがnan or inf の場合を除く
            if torch.isnan(loss) or torch.isinf(loss):
                net = prev_net
                optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
                optimizer.load_state_dict(prev_optimizer.state_dict())
            else:
                prev_net = copy.deepcopy(net)
                prev_optimizer = copy.deepcopy(optimizer)

                # 勾配をリセット
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # if i == 5000:
            #   break

        # 平均lossを計算
        avg_train_loss = train_loss / len(train_loader.dataset)

        # 訓練データのlossをログで出す
        print('Epoch [{}/{}], Loss: {loss}'.format(epoch+1, params.num_epochs, loss=avg_train_loss))

        train_loss_list.append(train_loss)


def train_supervised(net):
    with open('torch_input_vectors', 'rb') as f:
        torch_input_vectors = pickle.load(f)

    train_loader = DataLoader(torch_input_vectors, batch_size=params.batch_size, shuffle=True)
    test_loader = 

    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc_list = 0

        # train
        net.train()
        for i ,data in enumerate(train_loader):
            net = net.to(device)
            labels = 
            labels = labels.to(device)
            optim.zero_grad()
            outputs = net(data)
            loss = super_criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += ()
            loss.backward()
            optim.step()

    net.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for data in test_loader:
            data = data.to(device)
            labels = data.label
            labels = labels.to(device)

            outputs = net(data)
            test_acc +=
            total += 
        print('精度: {} %'.format(100 * test_acc / total))






if __name__ == '__main__':
    # numpy2tensor()
    # net = Autoencoder(params.input_size, params.hidden_size, params.unsupervised_output_size)
    # net = net.to(device)

    net = SupervisedTrain(params.batch_size, params.input_size, params.hidden_size, params.output)
    net = net.to(device)

    # 複数GPU使用宣言
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        torch.backends.cudnn.benchmark = True
    # train_model(net)
    train_supervised(net)
