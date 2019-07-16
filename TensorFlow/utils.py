#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle as pickle
import numpy as np
import pandas as pd


class TextLoader(object):
    def __init__(self, batch_size,tag = None):
        self.tag = tag
        self.data_path = './data/' + self.tag + '/' + self.tag + '_train.csv'
        self.batch_size = batch_size
        self.seq_length = 700
        self.encoding = 'utf8'
        with open( './data/covab.pkl', 'br') as f:
            vocab = pickle.load(f)
        self.vocab = vocab
        self.vocab_size = len(vocab) + 1
        self.load_preprocessed(self.data_path)
        self.shuff()

    def load_preprocessed(self,data_path):
        if self.tag == 'inflammation':
            data = pd.read_csv(data_path)
            number_sick = len(data[data.label == 0])
            sick_indices = np.array(data[data.label == 0].index)

            normal_indices = data[data.label == 1].index
            random_normal_indices = np.random.choice(normal_indices, number_sick, replace=False)
            random_normal_indices = np.array(random_normal_indices)
            under_sample_indices = np.concatenate([sick_indices, random_normal_indices])
            under_sample_data = data.iloc[under_sample_indices, :]
            X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'label']
            y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'label']
            tensor_x = np.array(X_undersample)
            tensor_y = np.array(y_undersample)
        else:
            data = pd.read_csv(data_path)
            X = data.ix[:, data.columns != 'label']
            Y = data.ix[:, data.columns == 'label']
            number_sick = len(data[data.label == 1])
            sick_indices = np.array(data[data.label == 1].index)
            normal_indices = data[data.label == 0].index
            random_normal_indices = np.random.choice(normal_indices, number_sick, replace=False)

            random_normal_indices = np.array(random_normal_indices)
            under_sample_indices = np.concatenate([sick_indices, random_normal_indices])

            under_sample_data = data.loc[under_sample_indices, :]

            X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'label']
            y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'label']
            tensor_x = np.array(X_undersample)
            tensor_y = np.array(y_undersample)
            self.tensor = np.c_[tensor_x, tensor_y].astype(int)
            # self.tensor = np.c_[X, Y].astype(int)

    def shuff(self):
        self.num_batches = int(self.tensor.shape[0] // self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        np.random.shuffle(self.tensor)

    def next_batch(self,k):
        x = []
        y = []
        for i in range(self.batch_size):
            tmp = np.array(list(self.tensor)[k*self.batch_size + i][:self.seq_length])
            x.append(tmp)
            tmp2 = np.array(list(self.tensor)[k*self.batch_size + i][-1])
            y.append(tmp2)
        return np.array(x), np.array(y)

class TextLoader_test(object):
    def __init__(self,batch_size,tag):
        self.tag = tag
        self.data_path = './data/' + self.tag + '/' + self.tag + '_test.csv'
        self.batch_size = batch_size
        self.seq_length = 700
        self.encoding = 'utf8'
        with open('./data/covab.pkl', 'br') as f:
            vocab = pickle.load(f)
        self.vocab = vocab
        self.vocab_size = len(vocab) + 1
        self.load_preprocessed(self.data_path)
        self.shuff()
    def load_preprocessed(self,data_path):
        data = pd.read_csv(data_path)
        X = data.ix[:, data.columns != 'label']
        Y = data.ix[:, data.columns == 'label']
        tensor_x = np.array(X)
        tensor_y = np.array(Y)
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)
    def shuff(self):
        self.num_batches = int(self.tensor.shape[0] // self.batch_size)
        np.random.shuffle(self.tensor)
    def next_batch(self,k):
        x = []
        y = []
        for i in range(self.batch_size):
            tmp = np.array(list(self.tensor)[k*self.batch_size + i][:self.seq_length])
            x.append(tmp)
            tmp2 = np.array(list(self.tensor)[k*self.batch_size + i][-1])
            y.append(tmp2)
        return np.array(x), np.array(y)


if __name__ == "__main__":
    data_loader = TextLoader_test(2,'liver')
    print(data_loader.tensor.shape)
    k = 0
    for i in data_loader.tensor:
        if i[700] == 1:
            k+=1
    print(k)
    x,y = data_loader.next_batch(4)
    print(y)
    # print(data_loader.tensor.shape)
    # a,b = data_loader.next_batch(3)
    # print(a.shape)
    # print(a)
    # print(b)
    # c,d = data_loader.next_batch(1)
    # print(c.shape)
    # print(d)



