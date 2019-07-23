import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable

from torchtext import data
from torchtext import vocab
from torchtext import datasets

import numpy as np
from matplotlib import pyplot as plt
import math

#CPU, GPUの使用設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''ハイパーパラメータの設定'''
batch_size = 32 #バッチサイズ
output_size = 8 #fcの出力先
hidden_size = 256 #lstmの出力次元
embedding_length = 96 #埋め込みベクトルの出力次元


'''前処理用の機能のFieldをセットアップ'''
'''トークン化の方式を指定'''
tokenize = lambda x: x.split()

'''
# Field
include_length=Trueでイテレータが長さも含めたタプルを返す
batch_first=Trueにしている
'''
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=False, include_lengths=True, batch_first=True, fix_length=200)

# LabelField
LABEL = data.LabelField()
print(LABEL)

train_dataset, test_dataset = datasets.IMDB.splits(TEXT, LABEL)
train_dataset, val_dataset = train_dataset.split()

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

# '''build_vocabで単語の辞書を構築．IDと単語の一覧が作成される'''
# '''学習済みベクトルに適応'''
# ''' min_freqで出現頻度の低い単語を省く'''
# TEXT.build_vocab(train_dataset, min_freq=3, vectors=vocab.GloVe(name='6B', dim=300))
# LABEL.build_vocab(train_dataset)

# #単語の件数top10
# print(TEXT.vocab.freqs.most_common(10))
# #ラベルごとの件数
# print(LABEL.vocab.freqs)
# #単語
# print(TEXT.vocab.itos[:10])
# #BucketItretorでバッチ単位にする
# train_iter, val_iter, test_iter = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

# #単語数
# vocab_size = len(TEXT.vocab)
# #単語数のサイズ
# print(vocab_size)
# #埋め込みベクトル
# word_embeddings = TEXT.vocab.vectors
# #埋め込みベクトルのサイズ
# print(TEXT.vocab.vectors.size())

# # データの確認
# for i, batch in enumerate(train_iter):
#     #バッチサイズ, 単語列の長さ
#     #バッチサイズ32, data.Fieldでfix_length=200を指定したので200文字
#     print(batch.text[0].size())

#     #ラベル
#     print(batch.text[1].size())
#     print(batch.label.size())
#     print("1データ目の単語列を表示")
#     print(batch.text[0][0])
#     print(batch.text[1][0])
#     print(batch.label[0])
#     print([TEXT.vocab.itos[data] for data in batch.text[0][0].tolist()])
#     print("ラベル")
#     print(batch.label[0].item())
#     break

# class LstmClassifier(nn.Module):
#     def __init__(self, batch_size, hidden_size, output_size, vocab_size, embedding_length, weights):
#         super(LstmClassifier, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.vocab_size = vocab_size
#         self.embed = nn.Embedding(vocab_size, embedding_length)
#         #学習済みの埋め込みベクトルを使用
#         self.embed.weight.data.copy_(weights)
#         self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.embed(x)
#         #初期隠れ状態とセル状態を設定
#         h0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
#         c0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
#         #LSTMを伝播する
#         #output_seqの出力形状: (バッチサイズ, シーケンス長, 出力次元)
#         output_seq, (h_n, c_n) = self.lstm(x, (h0, c0))
#         #最後のタイムステップの隠れ状態をデコード
#         out = self.fc(h_n[-1])
#         return out

# net = LstmClassifier(batch_size, hidden_size, output_size, vocab_size, embedding_length, word_embeddings)
# net = net.to(device)

# #損失関数，最適化関数を定義
# criterion = nn.CrossEntropyLoss()
# optim = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))

# num_epochs = 10

# train_loss_list = []
# train_acc_list = []
# val_loss_list = []
# val_acc_list = []

# for epoch in range(num_epochs):
#     train_loss = 0
#     train_acc = 0
#     val_loss = 0
#     val_acc = 0

#     #train
#     net.train()
#     for i, batch in enumerate(train_iter):
#         text = batch.text[0]
#         text = text.to(device)
#         if (text.size()[0] is not 32):
#             continue
#         labels = batch.label
#         labels = labels.to(device)
#         optim.zero_grad()
#         outputs = net(text)
#         loss = criterion(outputs, labels)
#         train_loss += loss.item()
#         train_acc += (outputs.max(1)[1] == labels).sum().item()
#         loss.backward()
#         optim.step

#     net.eval()
#     with torch.no_grad():
#         total = 0
#         test_acc = 0
#         for batch in test_iter:
#             text = batch.text[0]
#             text = text.to(device)
#             if (text.size()[0] is not 32):
#                 continue
#             labels = batch.label
#             labels = labels.to(device)

#             outputs = net(text)
#             test_acc += (outputs.max(1)[1] == labels).sum().item()
#             total += labels.size(0)

#         print('精度: {}%'.format(100 * test_acc / total))