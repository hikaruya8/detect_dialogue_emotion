# -*- coding:utf-8 -*-

import re
import os
import argparse
import pickle
from tqdm import tqdm 
from itertools import chain
import numpy as np
import glob
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import logging

# ターミナルでパラメータを変えられるようにする
parser = argparse.ArgumentParser()
parser.add_argument("--sg", type=int, default=1)
parser.add_argument("--size", type=int, default=96)
parser.add_argument("--window", type=int, default=5)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--min_count", type=int, default=5)
params = parser.parse_args()
print(params)


# primary emotion vectors
joy = np.array([1, 0, 0, 0])
sadness = np.array([-1, 0, 0, 0])
anger = np.array([0, 1, 0, 0])
fear = np.array([0, -1, 0 ,0])
trust = np.array([0, 0, 1, 0])
disgust = np.array([0, 0, -1, 0])
anticipation = np.array([0, 0, 0, 1])
surprise = np.array([0, 0, 0, -1])
primary_emo_vectors = [joy, sadness, anger, fear, trust, disgust, anticipation, surprise]

# tokenizerを定義
tokenizer = RegexpTokenizer("[\w']+|[\_]")

# mov_list内のファイルパスを取得
mov_path = glob.glob('../data/mov_list/*')
# word2vecのモデルの保存先
my_word2vec = 'my_word2vec.model'


def collect_data(mov_path):
    sentences = []
    for i, mp in enumerate(tqdm(mov_path)):
        texts = [tokenizer.tokenize(line) for line in open(mp, encoding='ISO-8859-1')]
        actor = [a[0] for a in texts if a!=[]]
        utterance = [u[1:] for u in texts if u!=[]]
        sentences.append(utterance)
    sentences = list(chain.from_iterable(sentences))
    words = list(chain.from_iterable(sentences))

    with open('sentences.pickle', 'wb') as f:
        pickle.dump(sentences, f)
    with open('words.pickle', 'wb') as f:
        pickle.dump(words, f)
    # with open('sentences.pickle', 'rb') as f:
    #     sentences = pickle.load(f)
    # with open('words.pickle', 'rb') as f:
    #     words = pickle.load(f)

    # print(sentences[0])
    # print(words[0])


def save_word2vec():
    with open('sentences.pickle', 'rb') as f:
        sentences = pickle.load(f)
    #学習してword2vecを構築. ./my_word2vec.modelファイルに保存
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences, sg=params.sg, size=params.size, window=params.window, iter=params.iter, min_count=params.min_count)
    model.save('./my_word2vec.model')


def load_word2vec(data_path):
    with open('words.pickle', 'rb') as f:
        words = pickle.load(f)
    #学習したword2vecをロード
    model = Word2Vec.load(data_path)

    try:
        word_vectors = np.array([model.wv[word] for i,word in enumerate(tqdm(words)) if word in words])
    except KeyError:
        pass

    with open('word2vec.pickle', 'wb') as f:
        pickle.dump(word_vectors, f, protocol=4)
 
    # word2vecが変な学習をしていないか確認(codeとして間違えてないか確認)
    print(model.wv.similarity('joy', 'sadness'))
    print('\n')

    results = model.wv.most_similar(positive=['joy'])
    for result in results:
        print(result)


def emo2vec(data_path):
    with open('words.pickle', 'rb') as f:
        words = pickle.load(f)
    # emotional words to vectors
    emo_vectors = np.empty((0,4), float)
    model = Word2Vec.load(data_path)
    primary_emo = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'anticipation', 'surprise']

    # make list of each cosine similarity
    each_sim = [model.wv.similarity(word, pe).astype(np.float32) for pe in tqdm(primary_emo, desc='2nd loop') for i,word in enumerate(tqdm(words, desc='1st loop'))]
    with open('each_sim.pickle', 'wb') as f:
        pickle.dump(each_sim, f, protocol=4)
    # make list of emotion vectors
    emo_vectors = [(np.sum([s*p for i, (s, p) in enumerate(tqdm(zip(each_sim, primary_emo_vectors), desc='4th loop'))], axis=0))/8 for i, word in enumerate(tqdm(words, desc='3rd loop'))]

    # for i, word in enumerate(tqdm(words)):
    #     each_sim = [model.wv.similarity(word, pe).astype(np.float32) for i, pe in enumerate(primary_emo)]
    #     each_word_vector = np.sum([s*p for i, (s, p) in enumerate(zip(each_sim, primary_emo_vectors))], axis=0)
    #     emo_vectors = (np.append(emo_vectors, [each_word_vector], axis=0)) / 8
    #     # if i == 100:
    #     #     break
    # print(emo_vectors[:10])
    with open('emo2vec.pickle', 'wb') as f:
        pickle.dump(emo_vectors, f, protocol=4) 


def final_vectors():
    # word_vectorsをファイルからロードする
    with open('word2vec.pickle', 'rb') as f:
        word_vectors = pickle.load(f)
    # emo_vectorsをファイルからロードする
    with open('emo2vec.pickle', 'rb') as f:
        emo_vectors = pickle.load(f)
    # wordsをファイルからロードする
    with open('words.pickle', 'rb') as f:
        words = pickle.load(f)

    # print(word_vectors[0])
    # print(emo_vectors[0])

    # 合わせて100次元のベクトルを作成
    final_vectors = [np.concatenate([wv, e]) for i, (wv, e) in enumerate(tqdm(zip(word_vectors, emo_vectors), total=len(word_vectors)))] 
    # final_vectors_dict = {w:f for w, f in zip(words, final_vectors)}

    with open('final_vectors.pickle', 'wb') as f:
        pickle.dump(final_vectors, f, protocol=4)

    # with open('final_vectors_dict.pickle', 'wb') as f:
    #     pickle.dump(final_vectors_dict, f)


def text2vec():
    # 全体の会話, 前の会話, 現在の会話をconcat
    with open('final_vectors.pickle', 'rb') as f:
        final_vectors = pickle.load(f)
    # with open('final_vectors_dict.pickle', 'rb') as f:
    #     final_vectors_dict = pickle.load(f)
    with open('sentences.pickle', 'rb') as f:
        sentences = pickle.load(f)

    # average of whole conversational vector
    whole_conv_vector = np.sum(final_vectors, axis=0) / len(sentences) 

    # get length of each sentence
    sentence_len = [len(s) for i,s in enumerate(tqdm(sentences, desc='1st loop'))]
    
    # # make list of current_vectors
    # sentence_len_sum = [sum(sentence_len[:i]) for i,s in enumerate(tqdm(sentence_len, desc='2nd loop'))]
    # with open('sentence_len_sum.pickle', 'wb') as f:
    #     pickle.dump(sentence_len_sum, f, protocol=4)

    with open('sentence_len_sum.pickle', 'rb') as f:
        sentence_len_sum = pickle.load(f)

    current_uttr_vectors = [(np.sum(final_vectors[sls:sls+1], axis=0))/sl for i,(sl, sls) in enumerate(tqdm(zip(sentence_len, sentence_len_sum), desc='3rd loop'))]


    # current_uttr_vectors = np.empty((0,100), float)
    # num = 0
    # for i,s in tqdm(enumerate(sentence_len)):
    #     current_uttr_vector = (np.sum(final_vectors[num:(num+s)], axis=0)) / s
    #     try:
    #         current_uttr_vectors = np.append(current_uttr_vectors, [current_uttr_vector], axis=0)
    #     except ValueError as e:
    #         pass
    #     num = num + s
    #     # if i == 50:
    #     #     break
    next_uttr_vectors = current_uttr_vectors[1:]

    # import pdb;pdb.set_trace()
    input_vectors = [np.concatenate((whole_conv_vector, c, n), axis=0) for i,(c,n) in enumerate(tqdm(zip(current_uttr_vectors[:-1], next_uttr_vectors), desc='4th loop'))]
    print(input_vectors[0:3])

    with open('input_vectors.pickle', 'wb') as f:
        pickle.dump(input_vectors, f, protocol=4)

 
def longest_uttr(data_path):
    # 最も長い発話を表示
    with open(data_path) as f:
        return len(max((x for x in f), key=len))


if __name__ == '__main__':
    # collect_data(mov_path)
    # save_word2vec()
    load_word2vec(my_word2vec)
    emo2vec(my_word2vec)
    final_vectors()
    text2vec()
