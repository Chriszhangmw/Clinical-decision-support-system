
import pickle as pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def splitsample(features,labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    # print('training set number is : ',len(features_train))
    # print('testing set number is : ',len(features_test))
    k = 0
    l = 0
    for e1 in labels_train:
        if e1 == 1:
            k +=1
    for e2 in labels_test:
        if e2 == 1:
            l +=1
    # print('negative samples in training set: ',k)
    # print('negative samples in testing set: ',l)
    return features_train, features_test, labels_train, labels_test

class TextLoader(object):
    def __init__(self, batch_size):
        self.data_path = 'D:\\Eclipse_workplace\\new_aad\\NLP_part\\word_embedding\\samples\\train.csv'
        self.batch_size = batch_size
        self.seq_length = 150
        self.encoding = 'utf8'
        self.load_preprocessed(self.data_path)
        self.shuff()

    def load_preprocessed(self,data_path):

        data = pd.read_csv(data_path)
        X = data.ix[:, data.columns != 'label']
        Y = data.ix[:, data.columns == 'label']
        number_sick = len(data[data.label == 1])
        sick_indices = np.array(data[data.label == 1].index)
        normal_indices = data[data.label == 0].index
        random_normal_indices = np.random.choice(normal_indices, number_sick + 300, replace=False)

        random_normal_indices = np.array(random_normal_indices)
        under_sample_indices = np.concatenate([sick_indices, random_normal_indices])

        under_sample_data = data.loc[under_sample_indices, :]

        X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'label']
        y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'label']
        tensor_x = np.array(X_undersample)
        tensor_y = np.array(y_undersample)
        features_train, features_test, labels_train, labels_test = splitsample(tensor_x,tensor_y)
        self.tensor = np.c_[features_train, labels_train].astype(int)
        self.testtensor = np.c_[features_test,labels_test].astype(int)
        self.num_batches = int(self.tensor.shape[0] // self.batch_size)
        self.test_num_batches = int(self.testtensor.shape[0] // self.batch_size)

    def shuff(self):
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
        # print('np.array(y).astype(int)',np.array(y).astype(int).shape)
        # print('np.array(y).astype(int)', np.array(y).astype(int))
        yy = np.zeros([self.batch_size,2])
        for i in range(len(y)):
            if y[i] == 1:
                yy[i][1] = 1
        # print(yy)
        return np.array(x).astype(int), np.array(yy).astype(float)

    def next_test_batch(self,kk):
        x = []
        y = []
        for i in range(self.batch_size):
            tmp = np.array(list(self.testtensor)[kk * self.batch_size + i][:self.seq_length])
            x.append(tmp)
            tmp2 = np.array(list(self.testtensor)[kk * self.batch_size + i][-1])
            y.append(tmp2)
        return np.array(x).astype(int), np.array(y).astype(int)

class TextLoader_vgg(object):
    def __init__(self, batch_size):
        self.data_path = 'D:\\Eclipse_workplace\\new_aad\\NLP_part\\word_embedding\\samples\\train_vgg.csv'
        self.batch_size = batch_size
        self.seq_length = 224
        self.encoding = 'utf8'
        self.load_preprocessed(self.data_path)
        self.shuff()

    def load_preprocessed(self,data_path):

        data = pd.read_csv(data_path)
        X = data.ix[:, data.columns != 'label']
        Y = data.ix[:, data.columns == 'label']
        tensor_x = np.array(X)
        tensor_y = np.array(Y)
        features_train, features_test, labels_train, labels_test = splitsample(tensor_x,tensor_y)
        self.tensor = np.c_[features_train, labels_train].astype(int)
        self.testtensor = np.c_[features_test,labels_test].astype(int)
        self.num_batches = int(self.tensor.shape[0] // self.batch_size)
        self.test_num_batches = int(self.testtensor.shape[0] // self.batch_size)

    def shuff(self):
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
        yy = np.zeros([self.batch_size,2])
        for i in range(len(y)):
            if y[i] == 1:
                yy[i][1] = 1
        # print(yy)
        return np.array(x).astype(int), np.array(yy).astype(float)

    def next_test_batch(self,kk):
        x = []
        y = []
        for i in range(self.batch_size):
            tmp = np.array(list(self.testtensor)[kk * self.batch_size + i][:self.seq_length])
            x.append(tmp)
            tmp2 = np.array(list(self.testtensor)[kk * self.batch_size + i][-1])
            y.append(tmp2)
        yy = np.zeros([self.batch_size, 2])
        for i in range(len(y)):
            if y[i] == 1:
                yy[i][1] = 1
        return np.array(x).astype(int), np.array(yy).astype(float)

